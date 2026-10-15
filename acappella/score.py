"""Read MuseScore (.mscz/.mscx) and MusicXML files into a simple timeline.

Positions are Fractions of a whole note, measured along the *playback*
order (repeats unrolled).  Score.time_map() turns them into seconds: for
MuseScore files on MuseScore's own tempo map (see musetempo.py), so the render
lines up with MuseScore's playback.
"""
from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from fractions import Fraction as F
from pathlib import Path

from . import musetempo

DUR = {
    "longa": F(4), "long": F(4), "breve": F(2), "whole": F(1), "half": F(1, 2),
    "quarter": F(1, 4), "eighth": F(1, 8), "16th": F(1, 16), "32nd": F(1, 32),
    "64th": F(1, 64), "128th": F(1, 128), "256th": F(1, 256), "1024th": F(1, 1024),
}
GRACE_TAGS = {"acciaccatura", "appoggiatura", "grace4", "grace8", "grace16", "grace32",
              "grace8after", "grace16after", "grace32after"}
DYNAMIC_LEVELS = ["pppppp", "ppppp", "pppp", "ppp", "pp", "p", "mp", "mf", "f", "ff",
                  "fff", "ffff", "fffff", "ffffff"]


@dataclass
class ChordEvent:
    start: F
    dur: F
    voice: int
    pitches: list            # MIDI numbers, highest first
    lyric: str | None = None
    syllabic: str | None = None  # single / begin / middle / end
    tie_next: set = field(default_factory=set)   # pitches tied into the next note
    gliss: set = field(default_factory=set)      # pitches that glide to the next note
    legato: bool = False      # slurred into the next chord
    staccato: float = 1.0     # fraction of written length actually sung
    accent: float = 0.0       # extra dB at the attack
    ghost: bool = False       # x-notehead / muted note: a short percussive hit
    measure: int = 0          # score measure index (for simile lookups)
    mpos: F = F(0)            # position inside the measure


@dataclass
class Staff:
    name: str
    instrument: str
    events: list = field(default_factory=list)       # ChordEvents (notes only)
    rests: list = field(default_factory=list)        # (start, dur, voice)
    dynamics: list = field(default_factory=list)     # (pos, subtype)
    hairpins: list = field(default_factory=list)     # (start, end, +1/-1)
    breaths: list = field(default_factory=list)      # positions
    percussion: bool = False


@dataclass
class Score:
    title: str
    staves: list
    tempos: list            # (pos, quarter-notes per minute)
    fermatas: list          # (pos, factor)
    end: F
    warnings: list = field(default_factory=list)
    measures: list = field(default_factory=list)   # (score measure no., play start, length)
    muse_tempo: object = None    # MuseScore's tempo map and repeat runs (MuseScore files)

    def time_map(self, bpm_override=None, tempo_scale=1.0):
        """Positions -> seconds.  --bpm and --tempo-scale act like MuseScore's
        playback speed: they scale the tempo but not pauses."""
        if self.muse_tempo is None:
            return TimeMap(self.tempos, self.fermatas, bpm_override=bpm_override,
                           tempo_scale=tempo_scale)
        tmap, runs = self.muse_tempo
        mult = tempo_scale
        if bpm_override:
            mult *= bpm_override / 60.0 / tmap.tempo(0)
        return musetempo.MuseTimeMap(tmap, runs, mult)


class TimeMap:
    """Whole-note positions -> seconds, honoring tempo changes and fermatas."""

    def __init__(self, tempos, fermatas, bpm_override=None, tempo_scale=1.0):
        tempos = sorted(tempos) or [(F(0), 120.0)]
        if tempos[0][0] > 0:
            tempos.insert(0, (F(0), tempos[0][1]))
        if bpm_override:
            first = tempos[0][1]
            tempos = [(p, q * bpm_override / first) for p, q in tempos]
        tempos = [(p, q * tempo_scale) for p, q in tempos]
        # breakpoints where the "seconds per whole note" rate changes
        pts = {p for p, _ in tempos}
        for p, d, _ in fermatas:
            pts.add(p)
            pts.add(p + d)
        self.pts = sorted(pts)
        self.tempos = tempos
        self.fermatas = fermatas
        self.cum = [0.0]
        for a, b in zip(self.pts, self.pts[1:]):
            self.cum.append(self.cum[-1] + float(b - a) * self._rate(a))

    def _rate(self, pos):
        q = self.tempos[0][1]
        for p, qq in self.tempos:
            if p <= pos:
                q = qq
        r = 240.0 / q
        for p, d, fac in self.fermatas:
            if p <= pos < p + d:
                r *= fac
        return r

    def sec(self, pos, before_pause=False) -> float:
        pos = F(pos)
        i = 0
        lo, hi = 0, len(self.pts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.pts[mid] <= pos:
                i = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if pos < self.pts[0]:
            return float(pos - self.pts[0]) * self._rate(self.pts[0])
        return self.cum[i] + float(pos - self.pts[i]) * self._rate(self.pts[i])


# ======================================================================
#  MuseScore
# ======================================================================
def _frac(s: str | None) -> F:
    if not s:
        return F(0)
    try:
        return F(s.strip())
    except (ValueError, ZeroDivisionError):
        return F(0)


def _loc(sp, which="next"):
    """Relative (measures, fraction) of a spanner's other end."""
    el = sp.find(which)
    if el is None:
        return None
    loc = el.find("location")
    if loc is None:
        return None
    return int(loc.findtext("measures") or 0), _frac(loc.findtext("fractions"))


def _text(el) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _duration(el) -> F:
    dt = el.findtext("durationType") or "quarter"
    if dt == "measure":
        d = _frac(el.findtext("duration")) or F(1)
    else:
        d = DUR.get(dt, F(1, 4))
    dots = int(el.findtext("dots") or 0)
    return d * (2 - F(1, 2 ** dots))


def _layout_breaks(el, section_pause: float):
    """(break kinds, section-break pause) of a measure or frame."""
    kinds, pause = set(), 0.0
    for lb in el.findall("LayoutBreak"):
        sub = (lb.findtext("subtype") or "").strip()
        kinds.add(sub)
        if sub == "section":
            pz = lb.findtext("pause")
            pause = float(pz) if pz else section_pause
    return kinds, pause


def _style_value(path: Path, sc, name: str, default: float) -> float:
    """A score style setting (score_style.mss in a .mscz, or <Style> in the score)."""
    found = None
    st = sc.find("Style")
    if st is not None:
        found = st.findtext(name)
    if found is None and path.suffix.lower() == ".mscz":
        try:
            with zipfile.ZipFile(path) as z:
                mss = [n for n in z.namelist() if n.endswith(".mss") and "/" not in n]
                if mss:
                    found = ET.fromstring(z.read(mss[0])).findtext(f"Style/{name}")
        except (zipfile.BadZipFile, ET.ParseError, KeyError):
            pass
    try:
        return float(found) if found is not None else default
    except ValueError:
        return default


def _read_mscz(path: Path) -> bytes:
    if path.suffix.lower() == ".mscx":
        return path.read_bytes()
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        root = None
        if "META-INF/container.xml" in names:
            c = ET.fromstring(z.read("META-INF/container.xml"))
            for rf in c.iter("rootfile"):
                fp = rf.get("full-path", "")
                if fp.endswith(".mscx"):
                    root = fp
                    break
        if root is None:
            cands = [n for n in names if n.endswith(".mscx") and "/" not in n]
            root = cands[0]
        return z.read(root)


def _voice_type_from_instrument(inst_id: str, name: str) -> str:
    s = f"{inst_id} {name}".lower()
    for key in ("mezzo", "soprano", "alto", "contralto", "countertenor", "tenor",
                "baritone", "bass"):
        if key in s:
            return {"mezzo": "mezzo-soprano", "contralto": "alto"}.get(key, key)
    return "voice"


def load_mscx(path: Path) -> Score:
    root = ET.fromstring(_read_mscz(path))
    sc = root.find("Score")
    if sc is None:
        raise ValueError("No <Score> element found")
    title = ""
    for mt in sc.findall("metaTag"):
        if mt.get("name") == "workTitle" and (mt.text or "").strip():
            title = mt.text.strip()
    if not title or title.lower().startswith("untitled"):
        title = path.stem

    # parts -> staff names in order
    staff_info = []
    for part in sc.findall("Part"):
        inst = part.find("Instrument")
        inst_id = inst.get("id", "") if inst is not None else ""
        name = (part.findtext("trackName") or (inst.findtext("longName") if inst is not None else "")
                or inst_id or "Voice").strip()
        drum = inst is not None and (inst.find("Drum") is not None
                                     or "drum" in inst_id or "percussion" in inst_id
                                     or (inst.findtext("useDrumset") or "0") == "1")
        n_st = max(1, len(part.findall("Staff")))
        for k in range(n_st):
            st_el = part.findall("Staff")[k] if part.findall("Staff") else None
            perc = drum
            if st_el is not None:
                stype = st_el.find("StaffType")
                if stype is not None and stype.get("group") == "percussion":
                    perc = True
            nm = name if n_st == 1 else f"{name} {k + 1}"
            staff_info.append((nm, inst_id, perc))

    staff_els = [s for s in sc.findall("Staff") if s.find("Measure") is not None]
    warnings: list[str] = []

    # ---- measure grid from the first staff
    timesig = F(1)
    m_len: list[F] = []
    m_pickup: list[bool] = []        # MuseScore's Measure::isAnacrusis()
    m_pause_before: list[float] = []  # section-break pause where the measure starts
    start_rep, end_rep, voltas = set(), {}, {}
    jumps = False
    section_pause = _style_value(path, sc, "SectionPause", 3.0)
    after_break, pending_pause = True, []    # first measure / after a frame or break
    mi = -1
    for m in staff_els[0]:
        if m.tag in ("VBox", "HBox", "TBox", "FBox"):
            pending_pause.append(_layout_breaks(m, section_pause)[1])
            after_break = True
            continue
        if m.tag != "Measure":
            continue
        mi += 1
        for v in m.findall("voice"):
            for e in v:
                if e.tag == "TimeSig":
                    n = int(e.findtext("sigN") or 4)
                    d = int(e.findtext("sigD") or 4)
                    timesig = F(n, d)
                if e.tag == "Spanner" and e.get("type") == "Volta" and e.find("Volta") is not None:
                    ends = e.find("Volta").findtext("endings") or "1"
                    nums = {int(x) for x in re.findall(r"\d+", ends)} or {1}
                    loc = _loc(e) or (1, F(0))
                    span = max(1, loc[0] + (1 if loc[1] > 0 else 0))
                    for k in range(span):
                        voltas[mi + k] = nums
                if e.tag in ("Jump", "Marker"):
                    jumps = True
            break  # only voice 1 carries these
        ln = m.get("len")
        m_len.append(_frac(ln) if ln else timesig)
        irregular = (m.findtext("irregular") or "0").strip() not in ("", "0")
        m_pickup.append((irregular or after_break) and timesig > m_len[-1])
        # MuseScore pauses at the earliest section break ending where this starts
        m_pause_before.append(next((x for x in pending_pause if x), 0.0))
        kinds, pz = _layout_breaks(m, section_pause)
        pending_pause = [pz]
        after_break = bool(kinds & {"line", "page", "section"})
        if m.find("startRepeat") is not None:
            start_rep.add(mi)
        if m.find("endRepeat") is not None:
            end_rep[mi] = int(m.findtext("endRepeat") or 2)
    if jumps:
        warnings.append("D.S./D.C./Coda jumps aren't followed; repeats are unrolled only")

    # ---- playback order
    order: list[int] = []
    i = 0
    rep_start = 0
    pass_no = 1
    counts: dict[int, int] = {}
    jumped = False
    guard = 0
    while i < len(m_len) and guard < 10000:
        guard += 1
        if i in start_rep:
            if not jumped:
                pass_no = 1
            rep_start = i
        jumped = False
        if i in voltas and pass_no not in voltas[i]:
            i += 1
            continue
        order.append(i)
        if i in end_rep:
            c = counts.get(i, 1)
            if c < end_rep[i]:
                counts[i] = c + 1
                pass_no = c + 1
                i = rep_start
                jumped = True
                continue
            rep_start = i + 1   # a later bare end-repeat goes back to here
        i += 1
    if any(k in end_rep for k in order) and len(order) > len(m_len):
        warnings.append(f"repeats unrolled: {len(m_len)} measures -> {len(order)} played")

    m_start_score = [F(0)]
    for ln in m_len:
        m_start_score.append(m_start_score[-1] + ln)
    occ: dict[int, list[F]] = {}
    pos = F(0)
    for mi in order:
        occ.setdefault(mi, []).append(pos)
        pos += m_len[mi]
    total = pos

    def play_positions(mi: int, rel: F):
        return [p + rel for p in occ.get(mi, [])]

    def score_abs(mi: int, rel: F) -> F:
        return m_start_score[min(mi, len(m_len))] + rel

    tempos: list[tuple] = []
    fermata_marks: list[tuple] = []   # (mi, rel, time stretch)
    tempo_marks: list[tuple] = []     # (mi, rel, quarter notes per second, type, follows text)
    breath_pauses: list[tuple] = []   # (mi, rel, seconds)
    tempo_lines: list[tuple] = []     # rit./accel.: (mi, rel, (measures, fraction), factor, easing)
    seg_points: set = set()           # (mi, rel) of notes, rests, breaths, clefs, barlines
    onset_points: set = set()         # (mi, rel) of notes and rests
    staves: list[Staff] = []

    for si, st in enumerate(staff_els):
        nm, inst_id, perc = staff_info[si] if si < len(staff_info) else (f"Staff {si + 1}", "", False)
        staff = Staff(name=nm, instrument=_voice_type_from_instrument(inst_id, nm), percussion=perc)
        slurs: list[tuple] = []     # (voice, score_start, score_end)
        raw_events: list[tuple] = []  # (mi, rel, ChordEvent, gliss/ties resolved later)

        for mi, m in enumerate(st.findall("Measure")):
            if mi >= len(m_len):
                break
            for vi, v in enumerate(m.findall("voice")):
                p = F(0)
                tup_stack: list[F] = []
                tup_by_id: dict[str, F] = {}
                for e in v:
                    tag = e.tag
                    scale = F(1)
                    for t in tup_stack:
                        scale *= t
                    if tag == "Tuplet":
                        ratio = F(int(e.findtext("normalNotes") or 2), int(e.findtext("actualNotes") or 3))
                        if e.get("id") is not None:      # MuseScore 3 style
                            tup_by_id[e.get("id")] = ratio
                        else:
                            tup_stack.append(ratio)
                    elif tag == "endTuplet":
                        if tup_stack:
                            tup_stack.pop()
                    elif tag == "location":
                        p += _frac(e.findtext("fractions"))
                    elif tag == "Tempo":
                        if (e.findtext("play") or "1").strip() == "0":
                            continue
                        try:
                            bps = float(e.findtext("tempo"))
                        except (TypeError, ValueError):
                            bps = musetempo.DEFAULT_TEMPO
                        tempos.append((mi, p, bps * 60.0))
                        follow = (e.findtext("followText") or "0").strip() not in ("", "0")
                        tempo_marks.append((mi, p, bps, (e.findtext("type") or "normal").strip(),
                                            follow))
                    elif tag in ("Clef", "KeySig", "TimeSig", "BarLine"):
                        seg_points.add((mi, p))
                    elif (tag == "Spanner" and e.get("type") == "GradualTempoChange"
                          and e.find("GradualTempoChange") is not None):
                        g = e.find("GradualTempoChange")
                        loc = _loc(e)
                        if loc and (g.findtext("play") or "1").strip() != "0":
                            kind = (g.findtext("tempoChangeType") or "undefined").strip()
                            fac = g.findtext("tempoChangeFactor")
                            factor = float(fac) if fac else musetempo.TEMPO_CHANGE_FACTOR.get(kind, 1.0)
                            easing = (g.findtext("tempoEasingMethod") or "normal").strip()
                            tempo_lines.append((mi, p, loc, factor, easing))
                    elif tag == "Dynamic":
                        sub = (e.findtext("subtype") or "").strip()
                        for pp in play_positions(mi, p):
                            staff.dynamics.append((pp, sub))
                    elif tag == "Spanner" and e.get("type") == "HairPin" and e.find("HairPin") is not None:
                        hp = e.find("HairPin")
                        sub = (hp.findtext("subtype") or "0").strip()
                        kind = -1 if sub in ("1", "3", "decrescendo", "diminuendo") else 1
                        loc = _loc(e)
                        if loc:
                            end_rel = score_abs(mi + loc[0], p + loc[1]) - score_abs(mi, F(0))
                            for pp in play_positions(mi, p):
                                staff.hairpins.append((pp, pp - p + end_rel, kind))
                    elif tag == "Breath":
                        for pp in play_positions(mi, p):
                            staff.breaths.append(pp)
                        seg_points.add((mi, p))
                        pz = e.findtext("pause")
                        if pz:
                            breath_pauses.append((mi, p, float(pz)))
                    elif tag == "Fermata":
                        # MuseScore's stretch: as set, or by the fermata's type
                        if (e.findtext("play") or "1").strip() == "0":
                            continue
                        ts = e.findtext("timeStretch")
                        fac = float(ts) if ts else musetempo.fermata_stretch(
                            e.findtext("subtype") or "fermataAbove")
                        fermata_marks.append((mi, p, fac))
                    elif tag == "Chord" or tag == "Rest":
                        if tag == "Chord" and any(e.find(g) is not None for g in GRACE_TAGS):
                            continue
                        tid = e.findtext("Tuplet")
                        if tid is not None and tid.strip() in tup_by_id:
                            scale *= tup_by_id[tid.strip()]
                        d = _duration(e) * scale
                        seg_points.add((mi, p))
                        onset_points.add((mi, p))
                        if tag == "Rest":
                            for pp in play_positions(mi, p):
                                staff.rests.append((pp, d, vi))
                            p += d
                            continue
                        ev = ChordEvent(start=F(0), dur=d, voice=vi, pitches=[], measure=mi, mpos=p)
                        for n in e.findall("Note"):
                            head = (n.findtext("head") or "").lower()
                            if (n.findtext("play") or "1").strip() == "0" or head in (
                                    "cross", "x", "xcircle", "withx", "slash", "ghost"):
                                ev.ghost = True
                            try:
                                pitch = int(n.findtext("pitch"))
                            except (TypeError, ValueError):
                                continue
                            pitch += int(n.findtext("tuning") or 0) / 100.0 if n.findtext("tuning") else 0
                            ev.pitches.append(pitch)
                            for sp in n.findall("Spanner"):
                                if sp.get("type") == "Tie" and sp.find("next") is not None:
                                    ev.tie_next.add(pitch)
                                if sp.get("type") == "Glissando" and sp.find("next") is not None:
                                    ev.gliss.add(pitch)
                            if n.find("Tie") is not None and n.find("Tie").get("id") is not None:
                                ev.tie_next.add(pitch)   # MuseScore 3 style
                        ev.pitches.sort(reverse=True)
                        for ly in e.findall("Lyrics"):
                            if int(ly.findtext("no") or 0) != 0:
                                continue
                            ev.lyric = _text(ly.find("text"))
                            ev.syllabic = (ly.findtext("syllabic") or "single").strip()
                        for sp in e.findall("Spanner"):
                            if sp.get("type") == "Slur" and sp.find("next") is not None:
                                loc = _loc(sp)
                                if loc:
                                    slurs.append((vi, score_abs(mi, p), score_abs(mi + loc[0], p + loc[1])))
                        for sl in e.findall("Slur"):   # MuseScore 3 slur start
                            if sl.get("type") == "start":
                                slurs.append((vi, score_abs(mi, p), None))
                        for a in e.findall("Articulation"):
                            sub = (a.findtext("subtype") or a.get("name") or "").lower()
                            if "staccatissimo" in sub:
                                ev.staccato = min(ev.staccato, 0.3)
                            elif "staccato" in sub and "tenuto" in sub:
                                ev.staccato = min(ev.staccato, 0.75)
                            elif "staccato" in sub:
                                ev.staccato = min(ev.staccato, 0.5)
                            if "marcato" in sub:
                                ev.accent = max(ev.accent, 6.0)
                            elif "accent" in sub or "sforzato" in sub:
                                ev.accent = max(ev.accent, 4.0)
                        if ev.pitches:
                            raw_events.append((mi, p, ev))
                        p += d

        # slurs -> legato flags (score space)
        for mi, p, ev in raw_events:
            s_abs = score_abs(mi, p)
            for vi, a, b in slurs:
                if vi == ev.voice and a <= s_abs and (b is None or s_abs < b):
                    ev.legato = True
        # unroll
        for mi, p, ev in raw_events:
            for pp in play_positions(mi, p):
                e2 = ChordEvent(**{**ev.__dict__})
                e2.start = pp
                staff.events.append(e2)
        staff.events.sort(key=lambda e: (e.start, e.voice))
        staff.dynamics.sort()
        staves.append(staff)

    # tempos / fermatas -> playback space
    tempo_pts: list[tuple] = []
    for mi, p, q in tempos:
        for pp in play_positions(mi, p):
            tempo_pts.append((pp, q))
    tempo_pts = sorted(set(tempo_pts))
    ferm: list[tuple] = []
    for mi, p, fac in sorted(set(fermata_marks)):
        fac = max(1.0, fac)
        for pp in play_positions(mi, p):
            # length of the shortest note/rest starting here in any staff
            d = None
            for st_ in staves:
                for ev in st_.events:
                    if ev.start == pp:
                        d = ev.dur if d is None else min(d, ev.dur)
                for (rs, rd, _) in st_.rests:
                    if rs == pp:
                        d = rd if d is None else min(d, rd)
            if d:
                ferm.append((pp, d, fac))
    mlist = []
    pos = F(0)
    for mi in order:
        mlist.append((mi + 1, pos, m_len[mi]))
        pos += m_len[mi]
    muse = _muse_tempo(m_start_score, m_len, m_pickup, m_pause_before, order, tempo_marks,
                       fermata_marks, breath_pauses, tempo_lines, seg_points, onset_points)
    return Score(title=title, staves=staves, tempos=tempo_pts, fermatas=ferm, end=total,
                 warnings=warnings, measures=mlist, muse_tempo=muse)


def _muse_tempo(m_start, m_len, m_pickup, m_pause_before, order, tempo_marks, fermata_marks,
                breath_pauses, tempo_lines, seg_points, onset_points):
    """MuseScore's tempo map for this score and the runs of consecutive
    measures it is played in."""
    tk = musetempo.ticks

    def tick(mi, rel):
        return tk(m_start[min(mi, len(m_len))] + rel)

    measures = [(tk(m_start[k]), tk(m_len[k]), m_pickup[k]) for k in range(len(m_len))]
    onsets = {tick(mi, p) for mi, p in onset_points}
    marks, seen = [], set()
    for mi, p, bps, kind, follow in tempo_marks:
        key = (tick(mi, p), bps, kind, follow)
        if key not in seen:
            seen.add(key)
            bps = min(max(bps, musetempo.MIN_TEMPO), musetempo.MAX_TEMPO)
            marks.append(musetempo.TempoMark(key[0], bps, kind, follow, key[0] in onsets))
    fermatas: dict[int, float] = {}
    for mi, p, fac in fermata_marks:
        if p < m_len[mi]:                   # one on a barline doesn't stretch time
            t = tick(mi, p)
            fermatas[t] = max(fermatas.get(t, 0.0), fac)
    pauses: dict[int, float] = {}
    for k, pz in enumerate(m_pause_before):
        if pz:
            pauses[measures[k][0]] = pz
    breath: dict[int, float] = {}
    for mi, p, pz in breath_pauses:
        t = tick(mi, p)
        breath[t] = max(breath.get(t, 0.0), pz)
    pauses.update({t: pz for t, pz in breath.items() if pz})
    changes = []
    for mi, p, loc, factor, easing in tempo_lines:
        t0 = tick(mi, p)
        changes.append(musetempo.TempoChange(t0, tick(mi + loc[0], p + loc[1]) - t0, factor, easing))
    seg_ticks = sorted({tick(mi, p) for mi, p in seg_points} | {s + n for s, n, _ in measures})
    tmap = musetempo.build_tempo_map(measures, marks, fermatas, pauses, changes, seg_ticks)
    runs = []
    pos, prev = F(0), None
    for mi in order:
        if runs and mi == prev + 1:
            u0, t0, n = runs[-1]
            runs[-1] = (u0, t0, n + measures[mi][1])
        else:
            runs.append((pos, measures[mi][0], measures[mi][1]))
        prev = mi
        pos += m_len[mi]
    return tmap, runs


# ======================================================================
#  MusicXML (via music21)
# ======================================================================
def load_musicxml(path: Path) -> Score:
    import music21 as m21  # type: ignore
    s = m21.converter.parse(str(path))
    try:
        s = s.expandRepeats()
    except Exception:
        pass
    title = (s.metadata.title if s.metadata and s.metadata.title else path.stem)
    tempos = []
    for mm in s.recurse().getElementsByClass(m21.tempo.MetronomeMark):
        q = mm.getQuarterBPM()
        if q:
            tempos.append((F(mm.getOffsetInHierarchy(s)).limit_denominator(10000) / 4, float(q)))
    staves = []
    for part in s.parts:
        name = part.partName or "Voice"
        staff = Staff(name=name, instrument=_voice_type_from_instrument("", name))
        flat = part.flatten()
        for el in flat.notesAndRests:
            start = F(el.offset).limit_denominator(10000) / 4
            dur = F(el.quarterLength).limit_denominator(10000) / 4
            if el.isRest:
                staff.rests.append((start, dur, 0))
                continue
            if dur == 0:
                continue
            pitches = sorted([p.midi for p in el.pitches], reverse=True)
            ev = ChordEvent(start=start, dur=dur, voice=0, pitches=pitches)
            if el.lyrics:
                ly = el.lyrics[0]
                ev.lyric = ly.text
                ev.syllabic = ly.syllabic or "single"
            notes = el.notes if el.isChord else [el]
            for n in notes:
                if n.tie is not None and n.tie.type in ("start", "continue"):
                    ev.tie_next.add(n.pitch.midi)
            for a in el.articulations:
                nm = type(a).__name__.lower()
                if "staccatissimo" in nm:
                    ev.staccato = 0.3
                elif "staccato" in nm:
                    ev.staccato = 0.5
                elif "strongaccent" in nm:
                    ev.accent = 6.0
                elif "accent" in nm:
                    ev.accent = 4.0
            staff.events.append(ev)
        for sp in flat.spanners:
            name_ = type(sp).__name__
            els = sp.getSpannedElements()
            if not els:
                continue
            a = F(els[0].getOffsetInHierarchy(part)).limit_denominator(10000) / 4
            last = els[-1]
            b = F(last.getOffsetInHierarchy(part) + last.quarterLength).limit_denominator(10000) / 4
            if name_ == "Slur":
                b_start = F(last.getOffsetInHierarchy(part)).limit_denominator(10000) / 4
                for ev in staff.events:
                    if a <= ev.start < b_start:
                        ev.legato = True
            elif name_ in ("Crescendo", "Diminuendo"):
                staff.hairpins.append((a, b, 1 if name_ == "Crescendo" else -1))
        for d in flat.getElementsByClass(m21.dynamics.Dynamic):
            staff.dynamics.append((F(d.offset).limit_denominator(10000) / 4, d.value))
        for b in flat.getElementsByClass(m21.articulations.BreathMark):
            staff.breaths.append(F(b.offset).limit_denominator(10000) / 4)
        staff.events.sort(key=lambda e: e.start)
        staff.dynamics.sort()
        staves.append(staff)
    end = max((e.start + e.dur for st in staves for e in st.events), default=F(1))
    mlist = []
    try:
        for k, m in enumerate(s.parts[0].getElementsByClass(m21.stream.Measure)):
            mlist.append((m.number or k + 1, F(m.offset).limit_denominator(10000) / 4,
                          F(m.duration.quarterLength).limit_denominator(10000) / 4))
    except Exception:
        pass
    return Score(title=title, staves=staves, tempos=sorted(set(tempos)), fermatas=[], end=end,
                 measures=mlist)


def load_score(path) -> Score:
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".mscz", ".mscx"):
        return load_mscx(path)
    if suf in (".musicxml", ".xml", ".mxl"):
        return load_musicxml(path)
    raise ValueError(f"Unsupported file type: {suf} (use .mscz, .mscx, .musicxml or .mxl)")
