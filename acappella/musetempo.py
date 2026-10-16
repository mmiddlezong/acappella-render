"""MuseScore's tempo map, rebuilt the way MuseScore 4 builds it, so a render
lines up with MuseScore's playback.

This follows Score::rebuildTempoAndTimeSigMaps, TempoMap and RepeatList in
MuseScore 4.7: tempo marks (with "a tempo" / "tempo primo"), fermatas
stretching the time to the next segment by their type's factor, breath and
section-break pauses, rit./accel. lines laid down as stepped tempo curves
with MuseScore's factors and easings, the pickup-measure fix, and repeats
timed per run of consecutive measures.  Positions are MuseScore ticks
(480 per quarter note) on the score's own (not unrolled) timeline.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np

DIVISION = 480
WHOLE = 4 * DIVISION
DEFAULT_TEMPO = 2.0                 # quarter notes per second
MIN_TEMPO, MAX_TEMPO = 5.0 / 60, 999.0 / 60

# Fermata::propertyDefault(TIME_STRETCH), by symbol (Above/Below dropped)
FERMATA_STRETCH = {
    "fermata": 2.0, "fermataVeryShort": 1.25, "fermataShort": 1.5, "fermataShortHenze": 1.5,
    "fermataLong": 3.0, "fermataLongHenze": 3.0, "fermataVeryLong": 4.0,
}
# GradualTempoChange DEFAULT_FACTORS_MAP
TEMPO_CHANGE_FACTOR = {
    "accelerando": 1.33, "allargando": 0.75, "calando": 0.5, "lentando": 0.75, "morendo": 0.5,
    "precipitando": 1.15, "rallentando": 0.75, "ritardando": 0.75, "smorzando": 0.5,
    "sostenuto": 0.95, "stringendo": 1.5,
}


def ticks(pos: F) -> int:
    """Fraction::ticks(): whole notes -> ticks, rounded half up."""
    pos = F(pos)
    sgn = -1 if pos.numerator < 0 else 1
    return sgn * ((sgn * pos.numerator * WHOLE + pos.denominator // 2) // pos.denominator)


def fermata_stretch(subtype: str) -> float:
    s = subtype.strip()
    for side in ("Above", "Below"):
        if s.endswith(side):
            s = s[: -len(side)]
    return FERMATA_STRETCH.get(s, 2.0)


def _round_tempo(bps: float) -> float:
    """RealRound(bps, TEMPO_PRECISION = 6)."""
    return math.floor(bps * 1e6 + 0.5) / 1e6


def _easing(x: np.float32, method: str) -> np.float32:
    """easingFactor(): float in and out, with std::pow on doubles in between."""
    xd = float(x)
    if method == "ease-in":
        v = 1 - math.sqrt(1 - xd ** 2)
    elif method == "ease-out":
        v = math.sqrt(1 - float(x - np.float32(1)) ** 2)
    elif method == "ease-in-out":
        if xd < 0.5:
            v = (1.0 - math.sqrt(1 - float(np.float32(2) * x) ** 2)) / 2
        else:
            v = (math.sqrt(1.0 - float(np.float32(-2) * x + np.float32(2)) ** 2) + 1) / 2
    elif method == "exponential":
        v = xd if x == np.float32(1) else 1.0 - 2.0 ** float(np.float32(-10) * x)
    else:                                   # normal (linear)
        v = xd
    return np.float32(v)


def eased_curve(duration: int, steps: int, amplitude: float, method: str) -> dict[int, float]:
    """buildEasedValueCurve(): {tick offset: value change}, in MuseScore's
    float precision (offsets truncated to whole ticks, first one wins)."""
    out: dict[int, float] = {}
    if steps <= 0:
        return out
    step = np.float32(duration) / np.float32(steps)
    for i in range(steps + 1):
        key = int(np.float32(i) * step)
        if key not in out:
            out[key] = float(_easing(np.float32(i) / np.float32(steps), method)) * amplitude
    return out


class TempoMap:
    """MuseScore's TempoMap: tick -> (tempo, pause, fixed).  An entry that only
    carries a pause keeps the tempo before it."""

    def __init__(self):
        self.ev: dict[int, list] = {}       # tick -> [tempo, pause, fixed]
        self._keys: list[int] = []
        self.times: list[float] = []

    def _add_key(self, tick):
        bisect.insort(self._keys, tick)

    def find(self, tick: int) -> bool:
        return tick in self.ev

    def tempo(self, tick) -> float:
        i = bisect.bisect_right(self._keys, tick) - 1
        while i >= 0:
            e = self.ev[self._keys[i]]
            if e[2]:
                return e[0]
            i -= 1
        return DEFAULT_TEMPO

    def set_tempo(self, tick: int, bps: float):
        if bps <= 0:
            bps = 0.01
        if tick in self.ev:
            self.ev[tick][0] = bps
            self.ev[tick][2] = True
        else:
            self.ev[tick] = [bps, 0.0, True]
            self._add_key(tick)

    def set_pause(self, tick: int, secs: float):
        if tick in self.ev:
            self.ev[tick][1] = secs
        else:
            self.ev[tick] = [self.tempo(tick), secs, False]
            self._add_key(tick)

    def pause(self, tick) -> float:
        e = self.ev.get(tick)
        return e[1] if e else 0.0

    def clear_range(self, t1: int, t2: int):
        for k in [k for k in self._keys if t1 <= k < t2]:
            del self.ev[k]
        self._keys = [k for k in self._keys if not t1 <= k < t2]

    def normalize(self, multiplier: float = 1.0):
        time, tick, tempo = 0.0, 0, DEFAULT_TEMPO
        self.times = []
        for k in self._keys:
            e = self.ev[k]
            if not e[2]:
                e[0] = tempo
            time += (k - tick) / (DIVISION * tempo * multiplier)
            time += e[1]
            self.times.append(time)
            tick, tempo = k, e[0]

    def tick2time(self, tick, multiplier: float = 1.0) -> float:
        i = bisect.bisect_right(self._keys, tick) - 1
        if i < 0:
            return tick / (DIVISION * DEFAULT_TEMPO * multiplier)
        k = self._keys[i]
        if k == tick:
            return self.times[i]
        return self.times[i] + (tick - k) / (DIVISION * self.ev[k][0] * multiplier)


@dataclass
class TempoMark:
    tick: int
    bps: float
    kind: str = "normal"          # normal, aTempo, tempoPrimo
    follow: bool = False
    on_chord: bool = True         # on a note/rest (not a bare time position)


@dataclass
class TempoChange:                # a rit./accel. line
    tick: int
    length: int
    factor: float
    easing: str = "normal"


def build_tempo_map(measures, marks, fermatas, pauses, changes, segment_ticks):
    """measures: [(start tick, length in ticks, is_pickup)] in score order;
    marks: [TempoMark]; fermatas: {tick: stretch}; pauses: {tick: secs};
    changes: [TempoChange]; segment_ticks: sorted ticks where MuseScore has an
    active segment (notes/rests in any staff, breaths, clefs, barlines,
    measure ends)."""
    tm = TempoMap()
    tm.set_tempo(0, DEFAULT_TEMPO)
    tempo_primo = None
    by_tick: dict[int, list] = {}
    for mk in marks:
        by_tick.setdefault(mk.tick, []).append(mk)
    ticks_with_items = sorted(set(by_tick) | set(fermatas))
    for t, secs in sorted(pauses.items()):
        tm.set_pause(t, secs)
    for t in ticks_with_items:
        for mk in by_tick.get(t, []):
            if mk.kind == "normal" and tempo_primo is None:
                tempo_primo = _round_tempo(mk.bps)
            if mk.kind == "aTempo" and mk.follow:
                tm.set_tempo(t, tm.tempo(t))        # back to the tempo before the change
            elif mk.kind == "tempoPrimo" and mk.follow:
                tm.set_tempo(t, tempo_primo if tempo_primo is not None else DEFAULT_TEMPO)
            else:
                tm.set_tempo(t, _round_tempo(mk.bps))
        stretch = fermatas.get(t, 0.0)
        if stretch and abs(stretch - 1.0) > 1e-12:
            old = tm.tempo(t)
            tm.set_tempo(t, _round_tempo(old / stretch))
            i = bisect.bisect_right(segment_ticks, t)
            end = segment_ticks[i] if i < len(segment_ticks) else t + 1
            if not tm.find(end - 1):
                tm.set_tempo(end - 1, old)
    # rit./accel. lines, in order of where they start
    for ch in sorted(changes, key=lambda c: c.tick):
        cur = tm.tempo(ch.tick)
        new = cur * ch.factor
        steps = max(8, ch.length // DIVISION)
        for off, dv in eased_curve(ch.length, steps, new - cur, ch.easing).items():
            if not tm.find(ch.tick + off):
                tm.set_tempo(ch.tick + off, _round_tempo(cur + dv))
    # a pickup takes the tempo marked in the following measure
    for k, (start, length, pickup) in enumerate(measures):
        if not pickup or k + 1 >= len(measures):
            continue
        own = [mk for mk in marks if start <= mk.tick < start + length and mk.on_chord]
        if own:
            continue
        nstart, nlen, _ = measures[k + 1]
        nxt = sorted((mk for mk in marks if nstart <= mk.tick < nstart + nlen and mk.on_chord),
                     key=lambda mk: mk.tick)
        if nxt:
            tm.set_tempo(start, _round_tempo(nxt[0].bps))
    return tm


class MuseTimeMap:
    """Played (repeats unrolled) whole-note positions -> seconds, timed the way
    MuseScore's RepeatList does: each run of consecutive measures on the
    score's tempo map."""

    def __init__(self, tempo_map: TempoMap, runs, multiplier: float = 1.0):
        """runs: [(played start position, score start tick, length in ticks)]."""
        self.tm = tempo_map
        self.mult = multiplier
        tempo_map.normalize(multiplier)
        self.runs = []
        t = 0.0
        for u0, tick0, length in runs:
            ct = tempo_map.tick2time(tick0, multiplier)
            self.runs.append((ticks(u0), tick0, t - ct))
            t += tempo_map.tick2time(tick0 + length, multiplier) - ct
        self._u = [r[0] for r in self.runs]
        self._seen = {}

    def sec(self, pos, before_pause=False) -> float:
        # every part asks about the same few thousand positions
        key = (pos, before_pause)
        t = self._seen.get(key)
        if t is None:
            t = self._seen[key] = self._sec(pos, before_pause)
        return t

    def _sec(self, pos, before_pause):
        u = F(pos) * WHOLE
        if before_pause:            # a note ending here: before a pause at this tick
            i = max(0, bisect.bisect_left(self._u, u) - 1)
        else:
            i = max(0, bisect.bisect_right(self._u, u) - 1)
        u0, tick0, off = self.runs[i]
        t = tick0 + (u - u0)
        tt = int(t) if t.denominator == 1 else float(t)
        time = self.tm.tick2time(tt, self.mult) + off
        if before_pause and t.denominator == 1:
            time -= self.tm.pause(int(t))
        return time
