"""Turn a parsed score into what each singer actually does: sung syllables
(Units) with timings in seconds, plus a loudness curve per staff."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction as F

import numpy as np

from .phonemes import Multi, Phonemizer, SONORANT_LIKE, detect_language, has_hangul
from .score import Score, Staff, TimeMap
from .synth import CR, SungNote, Unit

DYN_DB = {"pppppp": -36, "ppppp": -33, "pppp": -30, "ppp": -26, "pp": -20, "p": -14,
          "mp": -9, "mf": -5, "f": 0, "ff": 4, "fff": 7, "ffff": 9, "fffff": 10,
          "ffffff": 11}
ACCENT_DYN = {"sfz": 6, "sf": 6, "sffz": 8, "fz": 6, "rfz": 5, "rf": 5, "sfp": 6,
              "sfpp": 6, "fp": 5}
SIM_RE = re.compile(r"^\(?\s*(sim\.?|simile)\s*\)?$", re.I)
EPS = F(1, 1024)


@dataclass
class Item:
    start: F
    dur: F
    pitch: float
    lyric: str | None
    syllabic: str | None
    tie: bool
    gliss: bool
    legato: bool
    staccato: float
    accent: float
    ghost: bool
    measure: int
    mpos: F
    syl: tuple | None = None
    extra: list = field(default_factory=list)   # more syllables on the same note
    sim: bool = False
    korean: bool = False

    @property
    def end(self):
        return self.start + self.dur


@dataclass
class Line:
    staff: Staff
    name: str
    units: list = field(default_factory=list)


def _lines_for_staff(staff: Staff):
    """Split a staff into one-pitch lines: one per voice, plus extra lines
    for divisi chords (highest note = first line)."""
    by_voice: dict[int, list] = {}
    for ev in staff.events:
        by_voice.setdefault(ev.voice, []).append(ev)
    # voice-0 lyrics, for lower voices written without their own
    v0 = {}
    for ev in by_voice.get(0, []):
        if ev.lyric:
            v0[ev.start] = ev
    lines = []
    for v in sorted(by_voice):
        evs = by_voice[v]
        width = max(len(e.pitches) for e in evs)
        for k in range(width):
            items = []
            for e in evs:
                if k >= len(e.pitches):
                    continue
                lyric, syllabic = e.lyric, e.syllabic
                if not lyric and v > 0 and e.start in v0:
                    lyric, syllabic = v0[e.start].lyric, v0[e.start].syllabic
                p = e.pitches[k]
                items.append(Item(start=e.start, dur=e.dur, pitch=p, lyric=lyric or None,
                                  syllabic=syllabic, tie=p in e.tie_next, gliss=p in e.gliss,
                                  legato=e.legato, staccato=e.staccato, accent=e.accent,
                                  ghost=e.ghost, measure=e.measure, mpos=e.mpos))
            label = staff.name
            if len(by_voice) > 1:
                label += f" v{v + 1}"
            if width > 1:
                label += f" div{k + 1}"
            lines.append((label, items))
    return lines


def _merge_ties(items: list[Item]) -> list[Item]:
    out: list[Item] = []
    for it in items:
        if out:
            p = out[-1]
            if p.tie and abs(p.pitch - it.pitch) < 0.5 and abs(p.end - it.start) <= EPS:
                p.dur = it.end - p.start
                p.tie = it.tie
                p.gliss = it.gliss
                p.legato = it.legato
                p.staccato = it.staccato
                continue
        out.append(it)
    return out


def _assign_korean(items: list[Item], phon: Phonemizer, breaths) -> None:
    """Korean lyrics are usually entered one syllable per note, without
    word hyphens, and Korean pronunciation changes across syllables (음악 is
    sung 으-막).  So Korean pieces are read as one running line: each piece
    links to the next unless a rest, a breath mark or staccato separates
    them."""
    idx = [k for k, it in enumerate(items) if it.lyric and phon.is_korean(it.lyric.strip())]
    if not idx:
        return
    bset = list(breaths)
    links = []
    for a, b in zip(idx, idx[1:]):
        # another (non-Korean) lyric in between breaks the run
        ok = not any(items[j].lyric and re.search(r"\w", items[j].lyric) for j in range(a + 1, b))
        for k in range(a, b):
            if not ok:
                break
            x, y = items[k], items[k + 1]
            if (y.start - x.end > EPS or x.staccato < 0.9
                    or any(abs(x.end - bb) <= F(1, 32) for bb in bset)):
                ok = False
                break
        links.append(ok)
    links.append(False)
    sylls = phon.korean_line([items[k].lyric.strip() for k in idx], links)
    for k, s in zip(idx, sylls):
        it = items[k]
        it.korean = True
        if isinstance(s, Multi):
            it.syl, it.extra = s[0], list(s[1:])
        else:
            it.syl = s


def _assign_syllables(items: list[Item], phon: Phonemizer, warnings: list, breaths=()):
    """Phonemize lyrics word by word, then fill in simile passages."""
    _assign_korean(items, phon, breaths)
    word: list[Item] = []

    def flush():
        if not word:
            return
        sylls = phon.word([w.lyric for w in word])
        for w, s in zip(word, sylls):
            if isinstance(s, Multi):
                w.syl, w.extra = s[0], list(s[1:])
            else:
                w.syl = s
        word.clear()

    for it in items:
        if not it.lyric or it.korean:
            continue
        txt = it.lyric.strip()
        if SIM_RE.match(txt):
            flush()
            it.sim = True
            continue
        if not re.search(r"\w", txt):
            continue
        sy = (it.syllabic or "single").lower()
        if sy in ("single", "begin"):
            flush()
        word.append(it)
        if sy in ("single", "end"):
            flush()
    flush()

    # simile: copy the syllable pattern from the bar before (same beat)
    by_pos = {}
    sim_on = False
    last_syl = None
    for it in items:
        if it.syl is not None and not it.sim:
            sim_on = False
        if it.sim:
            sim_on = True
        if sim_on and it.syl is None:
            src = None
            for back in (1, 2, 4):
                src = by_pos.get((it.measure - back, it.mpos))
                if src is not None:
                    break
            it.syl = src or last_syl
            it.sim = True
        if it.syl is not None:
            by_pos[(it.measure, it.mpos)] = it.syl
            last_syl = it.syl


def _units_for_line(items: list[Item], tm: TimeMap, breaths: list, offset: float,
                    warnings: list, name: str) -> list[Unit]:
    units: list[Unit] = []
    cur: Unit | None = None
    cur_end = None
    resung = 0
    breath_set = set(breaths)
    for k, it in enumerate(items):
        t0 = tm.sec(it.start) + offset
        t_end = tm.sec(it.end) + offset
        t1 = t0 + (t_end - t0) * it.staccato
        if any(abs(it.end - b) <= F(1, 32) for b in breath_set):
            t1 -= min(0.15, 0.25 * (t1 - t0))
        note = SungNote(t0=t0, t1=max(t0 + 0.02, t1), midi=it.pitch, gliss=it.gliss)
        contiguous = cur is not None and cur_end is not None and abs(it.start - cur_end) <= EPS \
            and cur.notes[-1].t1 >= t0 - 0.03
        prev_legato = k > 0 and items[k - 1].legato

        syl = it.syl
        if syl is not None and not syl[1] and not syl[2] and not syl[0]:
            syl = None            # "chi-i-ild": a repeated vowel is just held
        if syl is not None and not syl[1] and cur is not None and contiguous:
            # consonant-only piece: keep going on the same syllable
            son = [c for c in syl[2] if c in SONORANT_LIKE]
            obs = [c for c in syl[2] if c not in SONORANT_LIKE]
            if son:
                note.nucleus = son[0]
            cur.notes.append(note)
            if obs:
                cur.coda = list(cur.coda) + obs
            cur_end = it.end
            continue
        if syl is not None and not syl[1] and syl[2]:
            # a consonant-only piece after a rest (Japanese ん, a lone "-l"):
            # sing it on its own, as a hum or a short hit
            son = [c for c in syl[2] if c in SONORANT_LIKE]
            obs = [c for c in syl[2] if c not in SONORANT_LIKE]
            syl = ([], [son[0]], obs) if son else (obs, ["n"], ["*ghost*"])
        if syl is None or not syl[1]:
            if cur is not None and contiguous and not it.ghost:
                cur.notes.append(note)               # melisma
                cur_end = it.end
                continue
            if cur is not None:
                syl = (cur.onset, cur.nucleus, cur.coda)
                resung += 1
            else:
                syl = ([], ["u"], [])
                warnings.append(f"{name}: notes before the first lyric are sung on 'oo'")
        onset, nucleus, coda = syl
        ghost = "*ghost*" in coda or it.ghost
        cur = Unit(onset=list(onset), nucleus=list(nucleus),
                   coda=[c for c in coda if c != "*ghost*"], notes=[note],
                   accent=it.accent - (6.0 if it.ghost else 0.0), ghost=ghost,
                   legato_in=prev_legato and contiguous)
        units.append(cur)
        cur_end = it.end
        if it.extra and not it.sim:
            # share the note among the syllables written under it
            parts = 1 + len(it.extra)
            seg = (note.t1 - note.t0) / parts
            note.t1 = note.t0 + seg
            for j, (o2, n2, c2) in enumerate(it.extra):
                a2 = note.t0 + seg * (j + 1)
                n2note = SungNote(t0=a2, t1=a2 + seg, midi=it.pitch)
                cur = Unit(onset=list(o2), nucleus=list(n2), coda=list(c2), notes=[n2note],
                           legato_in=True)
                units.append(cur)
            cur.notes[-1].t1 = max(t1, cur.notes[-1].t0 + 0.02)
    if resung:
        warnings.append(f"{name}: {resung} lyric-less notes after a rest re-sing the previous syllable")
    return units


def gain_curve(staff: Staff, tm: TimeMap, n_frames: int, offset: float,
               default: str = "mf") -> tuple[np.ndarray, list]:
    """Loudness in dB at control rate from dynamics and hairpins."""
    events = []
    accents = []
    for pos, sub in staff.dynamics:
        s = sub.lower().strip()
        t = tm.sec(pos) + offset
        if s in DYN_DB:
            events.append((t, 0, "dyn", DYN_DB[s]))
        elif s in ACCENT_DYN:
            accents.append((t, ACCENT_DYN[s]))
            if s in ("fp", "sfp"):
                events.append((t, 0, "dyn", DYN_DB["f"]))
                events.append((t + 0.15, 0, "dyn", DYN_DB["p"]))
            if s == "sfpp":
                events.append((t + 0.15, 0, "dyn", DYN_DB["pp"]))
    for a, b, kind in staff.hairpins:
        events.append((tm.sec(a) + offset, 1, "hp", (tm.sec(b) + offset, kind, b)))
    events.sort(key=lambda e: (e[0], e[1]))
    dyn_at = sorted((tm.sec(p) + offset, DYN_DB[s.lower()]) for p, s in staff.dynamics
                    if s.lower() in DYN_DB)

    cur = DYN_DB.get(default, -5)
    anchor = cur              # last written dynamic
    pts = [(0.0, cur)]
    hp_until = -1.0
    for t, _o, kind, val in events:
        if kind == "dyn":
            if t < hp_until - 1e-3:
                anchor = val
                continue          # the hairpin already heads to it
            pts.append((t, cur))
            cur = val
            anchor = val
            pts.append((t, cur))
        else:
            tb, direction, _pos = val
            if tb <= t:
                continue
            target = None
            for td, lv in dyn_at:
                if tb - 0.12 <= td <= tb + 0.35:
                    target = lv
                    break
            if target is None or (direction > 0 and target <= cur) or (direction < 0 and target >= cur):
                # unmarked swell: one dynamic step, but don't drift far from
                # the last written dynamic
                target = float(np.clip(cur + 6 * direction, anchor - 9, anchor + 7))
                if abs(target - cur) < 2:
                    target = cur + 2 * direction
            target = float(np.clip(target, -30, 10))
            pts.append((t, cur))
            pts.append((tb, target))
            cur = target
            hp_until = tb
    pts.append((n_frames / CR + 1, cur))
    ts = np.array([p[0] for p in pts])
    vs = np.array([p[1] for p in pts], dtype=float)
    frames = np.arange(n_frames) / CR
    curve = np.interp(frames, ts, vs)
    return curve, accents


def voice_kind(staff: Staff, items) -> str:
    kind = staff.instrument
    if kind != "voice":
        return kind
    ps = sorted(p for ev in staff.events for p in ev.pitches)
    med = ps[len(ps) // 2] if ps else 60
    if med >= 64:
        return "mezzo-soprano"
    if med >= 58:
        return "alto"
    if med >= 52:
        return "tenor"
    return "baritone"


def _lyric_words(score: Score) -> list:
    """Lyric words as lists of their hyphenated pieces (for language
    detection)."""
    words = []
    for st in score.staves:
        cur: list = []
        for ev in sorted(st.events, key=lambda e: (e.voice, e.start)):
            if not ev.lyric:
                continue
            sy = (ev.syllabic or "single").lower()
            if sy in ("single", "begin") and cur:
                words.append(cur)
                cur = []
            cur.append(ev.lyric)
            if sy in ("single", "end"):
                words.append(cur)
                cur = []
        if cur:
            words.append(cur)
    return words


def plan(score: Score, tm: TimeMap, offset: float, lang: str, n_frames: int):
    """Returns (lines, gains, warnings, phonemizer)."""
    warnings = list(score.warnings)
    toks = [ev.lyric for st in score.staves for ev in st.events if ev.lyric]
    if lang == "auto":
        lang = detect_language(toks, _lyric_words(score))
    if lang in ("ko", "ko-rom"):
        # Hangul lyrics: Latin ones are vocables/English.  No Hangul at all:
        # Latin lyrics are romanized Korean.
        lang = "ko" if any(has_hangul(t) for t in toks) else "ko-rom"
    phon = Phonemizer(lang)
    out = []
    for st in score.staves:
        if st.percussion:
            warnings.append(f"{st.name}: percussion staff skipped")
            continue
        if not st.events:
            continue
        gain, accents = gain_curve(st, tm, n_frames, offset)
        for name, items in _lines_for_staff(st):
            items = _merge_ties(items)
            _assign_syllables(items, phon, warnings, st.breaths)
            units = _units_for_line(items, tm, st.breaths, offset, warnings, name)
            for t, db in accents:
                for u in units:
                    if abs(u.t_on - t) < 0.03:
                        u.accent = max(u.accent, db)
            out.append((st, name, units, gain))
    warnings += phon.warnings
    return out, warnings, lang
