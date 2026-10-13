"""A small Klatt-style formant singing synthesizer.

Each singer is rendered from control tracks sampled every millisecond:
pitch, voicing amplitude, aspiration amplitude, five formants with
bandwidths, and a nasal pole/zero pair.  A band-limited glottal pulse plus
noise goes through a cascade of resonators; fricatives and stop bursts are
filtered noise added on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt

try:  # scipy's compiled core, ~15x less overhead per call than sosfilt()
    from scipy.signal._sosfilt import _sosfilt as _sosfilt_fast
except Exception:  # pragma: no cover
    _sosfilt_fast = None

SR = 44100
CR = 1000.0          # control frames per second
BLOCK = 64           # samples per filter-coefficient update

# --------------------------------------------------------------- phonetics
# F1..F3 for an adult male voice; scaled per voice type.
VOWEL_F = {
    "i": (290, 2250, 2900), "I": (400, 1950, 2550), "e": (440, 2050, 2650),
    "E": (550, 1800, 2500), "ae": (680, 1700, 2450), "a": (720, 1150, 2550),
    "O": (580, 880, 2500), "o": (460, 820, 2450), "U": (450, 1050, 2350),
    "u": (320, 780, 2250), "V": (620, 1200, 2550), "@": (500, 1450, 2500),
    "3": (480, 1350, 1700), "y": (300, 1850, 2300), "1": (380, 1500, 2400),
    "M": (350, 1250, 2350),
}
VOWEL_B = (70.0, 90.0, 130.0, 170.0, 220.0)

PLACE = {}
for c in "b p m f v w".split():
    PLACE[c] = "lab"
for c in "T D".split():
    PLACE[c] = "dent"
for c in "d t n s z l ts dz R r".split():
    PLACE[c] = "alv"
for c in "S Z tS dZ C j".split():
    PLACE[c] = "post"
for c in "k g N x".split():
    PLACE[c] = "vel"
LOCUS = {"lab": (250, 850, 2200), "dent": (300, 1500, 2600), "alv": (280, 1700, 2600),
         "post": (280, 2050, 2700)}
SONORANT_F = {
    "m": (280, 1050, 2200), "n": (280, 1500, 2500), "N": (280, 1950, 2600),
    "l": (360, 1150, 2700), "r": (420, 1250, 1650), "R": (330, 1600, 2600),
    "w": (300, 650, 2200), "j": (270, 2150, 3000),
}
NASAL_ZERO = {"m": 1050, "n": 1650, "N": 2600}
STOP = {
    #      voiced  closure burst_fc burst_bw burst_amp burst_len aspiration
    "b": (True, 0.040, 1200, 1800, 0.10, 0.006, 0.0),
    "d": (True, 0.035, 4000, 3000, 0.22, 0.008, 0.0),
    "g": (True, 0.040, 2000, 1200, 0.25, 0.012, 0.0),
    "p": (False, 0.045, 1200, 2000, 0.25, 0.008, 0.040),
    "t": (False, 0.040, 4500, 3500, 0.40, 0.010, 0.040),
    "k": (False, 0.045, 2200, 1500, 0.40, 0.014, 0.045),
}
FRIC = {
    #      voiced  dur    fc     bw     amp
    "s": (False, 0.085, 6500, 4500, 0.30),
    "z": (True, 0.065, 6500, 4500, 0.16),
    "S": (False, 0.085, 3200, 2600, 0.34),
    "Z": (True, 0.065, 3200, 2600, 0.18),
    "C": (False, 0.085, 4200, 3000, 0.28),
    "f": (False, 0.070, 6000, 8000, 0.07),
    "v": (True, 0.055, 6000, 8000, 0.04),
    "T": (False, 0.060, 6000, 7000, 0.06),
    "D": (True, 0.045, 6000, 7000, 0.035),
}
AFFRIC = {"tS": ("t", "S"), "dZ": ("d", "Z"), "ts": ("t", "s"), "dz": ("d", "z")}
SONORANT_DUR = {"m": 0.065, "n": 0.060, "N": 0.065, "l": 0.055, "r": 0.065, "R": 0.022,
                "w": 0.060, "j": 0.055}
SONORANT_AV = {"m": 0.50, "n": 0.50, "N": 0.50, "l": 0.70, "r": 0.75, "R": 0.55,
               "w": 0.75, "j": 0.80}

PROFILES = {
    #               formant scale, F4, F5, vibrato rate
    "soprano":       (1.18, 4100, 4900, 5.6),
    "mezzo-soprano": (1.15, 4000, 4800, 5.4),
    "alto":          (1.13, 3900, 4700, 5.3),
    "countertenor":  (1.08, 3700, 4500, 5.3),
    "tenor":         (1.04, 3500, 4400, 5.4),
    "baritone":      (1.00, 3350, 4300, 5.2),
    "bass":          (0.96, 3200, 4200, 5.0),
}


# ---------------------------------------------------------------- the plan
@dataclass
class SungNote:
    t0: float
    t1: float
    midi: float
    gliss: bool = False
    nucleus: str | None = None     # sonorant sung on this note ('l' in "fu-l")


@dataclass
class Unit:
    """One sung syllable: onset consonants, a nucleus held across one or
    more legato notes, then coda consonants."""
    onset: list
    nucleus: list
    coda: list
    notes: list
    accent: float = 0.0
    ghost: bool = False
    legato_in: bool = False
    # filled in by the planner
    c_start: float = 0.0
    c_scale: float = 1.0
    eff_off: float = 0.0
    joined_next: bool = False
    next_vowel_initial: bool = False

    @property
    def t_on(self):
        return self.notes[0].t0

    @property
    def t_off(self):
        return self.notes[-1].t1


# ------------------------------------------------------- glottal wavetables
_TABLE_N = 2048
_TABLES = None


def _glottal_tables():
    """Band-limited glottal-flow-derivative tables for 3 voice qualities
    (soft/breathy, normal, pressed) x several harmonic limits."""
    global _TABLES
    if _TABLES is not None:
        return _TABLES
    hi = 1 << 15
    t = np.arange(hi) / hi
    out = {}
    for qi, (oq, tilt) in enumerate(((0.70, 0.35), (0.58, 0.05), (0.46, -0.15))):
        u = np.where(t < oq, (t / oq) ** 2 - (t / oq) ** 3, 0.0)
        U = np.fft.rfft(u)
        k = np.arange(len(U))
        D = U * (2j * np.pi * k)                       # differentiate (lip radiation)
        D[1:] *= k[1:] ** (-tilt)
        D[0] = 0
        full = None
        tabs = {}
        for h in (2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
            Dh = np.zeros(_TABLE_N // 2 + 1, dtype=complex)
            m = min(h, _TABLE_N // 2)
            Dh[1:m + 1] = D[1:m + 1]
            w = np.fft.irfft(Dh, _TABLE_N)
            if full is None:
                full = np.sqrt(np.mean(w ** 2))
            tabs[h] = (w / full).astype(np.float32)
        out[qi] = tabs
        amps = np.abs(D[1:2049])
        out[("amps", qi)] = (amps / (full * _TABLE_N / 2)).astype(np.float64)
    _TABLES = out
    return out


def _hann_smooth(x: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 1:
        return x
    w = np.hanning(frames + 2)[1:-1]
    w /= w.sum()
    pad = frames
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(xp, w, mode="same")[pad:-pad]


# ------------------------------------------------------------------ singer
class Singer:
    def __init__(self, kind: str, duration: float, seed: int = 0,
                 vibrato_cents: float = 15.0, breathiness: float = 0.06):
        self.kind = kind if kind in PROFILES else "baritone"
        self.fs, self.f4, self.f5, self.vib_rate = PROFILES[self.kind]
        self.dur = duration
        self.n = int(duration * CR) + 2
        self.rng = np.random.default_rng(seed)
        self.vibrato_cents = vibrato_cents
        self.breathiness = breathiness
        n = self.n
        neutral = self._vowel("@")
        self.F = np.tile(np.array(neutral[0], dtype=float)[:, None], (1, n))
        self.B = np.tile(np.array(neutral[1], dtype=float)[:, None], (1, n))
        self.AV = np.zeros(n)
        self.AH = np.zeros(n)
        self.NZ = np.full(n, 270.0)      # nasal zero; 270 == pole -> no nasality
        self.midi = np.full(n, np.nan)
        self.vib = np.zeros(n)          # vibrato depth in cents
        self.cents = np.zeros(n)        # scoops / glide offsets
        self.accent_db = np.zeros(n)
        self.fric: list[tuple] = []     # (t0, t1, fc, bw, amp, attack, release)
        self.dips: list[tuple] = []     # (t_center, width, depth)

    # ---------------------------------------------------------- helpers
    def _i(self, t: float) -> int:
        return int(np.clip(round(t * CR), 0, self.n - 1))

    def _span(self, a: float, b: float):
        return slice(self._i(a), max(self._i(a) + 1, self._i(b)))

    def _vowel(self, v: str):
        f1, f2, f3 = VOWEL_F.get(v, VOWEL_F["@"])
        s = self.fs
        return ((f1 * s, f2 * s, f3 * s, self.f4, self.f5), VOWEL_B)

    def _cons_formants(self, c: str, vowel: str):
        vf, vb = self._vowel(vowel)
        s = self.fs
        if c in SONORANT_F:
            f1, f2, f3 = SONORANT_F[c]
            b = (100.0, 160.0, 220.0, 250.0, 300.0) if c in NASAL_ZERO else (80.0, 110.0, 160.0, 230.0, 280.0)
            return (f1 * s, f2 * s, f3 * s, self.f4, self.f5), b
        place = PLACE.get(c)
        if place == "vel":
            f2 = 0.55 * vf[1] + 900 * s
            return (260 * s, f2, f2 + 250 * s, self.f4, self.f5), (90.0, 110.0, 160.0, 230.0, 280.0)
        if place in LOCUS:
            f1, f2, f3 = LOCUS[place]
            # loci pull only part of the way from the vowel (coarticulation)
            return ((f1 * s, 0.35 * vf[1] + 0.65 * f2 * s, 0.4 * vf[2] + 0.6 * f3 * s,
                     self.f4, self.f5), (90.0, 110.0, 160.0, 230.0, 280.0))
        return vf, vb

    def _set_formants(self, a, b, fv):
        sl = self._span(a, b)
        f, bw = fv
        for k in range(5):
            self.F[k, sl] = f[k]
            self.B[k, sl] = bw[k]

    def _ramp(self, arr, a, b, v0, v1):
        sl = self._span(a, b)
        m = sl.stop - sl.start
        arr[sl] = np.linspace(v0, v1, m) if m > 1 else v1

    @staticmethod
    def cons_dur(c: str) -> float:
        if c in STOP:
            v, clo, *_r, asp = STOP[c]
            return clo + asp
        if c in FRIC:
            return FRIC[c][1]
        if c in AFFRIC:
            st, fr = AFFRIC[c]
            return STOP[st][1] * 0.8 + FRIC[fr][1] * 0.6
        if c in SONORANT_DUR:
            return SONORANT_DUR[c]
        if c in ("h", "x"):
            return 0.060
        return 0.04

    # ------------------------------------------------- consonant writers
    def _burst(self, t, c, amp_scale=1.0):
        voiced, clo, fc, bw, amp, blen, asp = STOP[c]
        fc = fc * (1.1 if self.fs > 1.08 else 1.0)
        self.fric.append((t, t + blen + 0.004, fc, bw, amp * amp_scale, 0.0005, blen))

    def _onset_cons(self, c, a, b, vowel):
        """Write consonant c occupying [a, b) before a vowel."""
        d = b - a
        if c in STOP:
            voiced, clo, fc, bw, amp, blen, asp = STOP[c]
            frac_clo = clo / (clo + asp) if asp else 1.0
            tb = a + d * frac_clo
            self._set_formants(a, tb, self._cons_formants(c, vowel))
            if voiced:
                self.AV[self._span(a, tb)] = 0.07
                self.F[0, self._span(a, tb)] = 180 * self.fs
            else:
                self.AV[self._span(a, tb)] = 0.0
            self.AH[self._span(a, tb)] = 0.0
            self._burst(tb, c)
            if asp:
                self._set_formants(tb, b, self._vowel(vowel))
                self.AV[self._span(tb, b)] = 0.0
                self._ramp(self.AH, tb, b, 0.55, 0.25)
        elif c in FRIC:
            voiced, _d, fc, bw, amp = FRIC[c]
            self._set_formants(a, b, self._cons_formants(c, vowel))
            self.AV[self._span(a, b)] = 0.35 if voiced else 0.0
            self.AH[self._span(a, b)] = 0.0
            self.fric.append((a, b, fc * (1.08 if self.fs > 1.08 else 1.0), bw, amp,
                              min(0.015, d * 0.3), min(0.012, d * 0.3)))
        elif c in AFFRIC:
            st, fr = AFFRIC[c]
            m = a + d * 0.45
            voiced = STOP[st][0]
            self._set_formants(a, m, self._cons_formants(fr, vowel))
            self.AV[self._span(a, m)] = 0.07 if voiced else 0.0
            self.AH[self._span(a, m)] = 0.0
            self._burst(m, st, 0.6)
            vf, _d, fc, bw, amp = FRIC[fr]
            self._set_formants(m, b, self._cons_formants(fr, vowel))
            self.AV[self._span(m, b)] = 0.3 if voiced else 0.0
            self.fric.append((m, b, fc, bw, amp * 1.1, 0.002, min(0.012, (b - m) * 0.4)))
        elif c in SONORANT_F:
            self._set_formants(a, b, self._cons_formants(c, vowel))
            self.AV[self._span(a, b)] = SONORANT_AV[c]
            self.AH[self._span(a, b)] = 0.0
            if c in NASAL_ZERO:
                self.NZ[self._span(a, b)] = NASAL_ZERO[c] * self.fs
        elif c in ("h", "x"):
            self._set_formants(a, b, self._vowel(vowel))
            self.AV[self._span(a, b)] = 0.0
            self._ramp(self.AH, a, b, 0.25, 0.55)
            if c == "x":
                self.fric.append((a, b, 1600 * self.fs, 1200, 0.06, 0.01, 0.01))

    def _coda_cons(self, c, a, b, vowel, final: bool, cluster: bool = False):
        """Write consonant c occupying [a, b) after a vowel.  `final` means
        the sound stops at b (rather than running into the next syllable)."""
        d = b - a
        if c in STOP:
            voiced = STOP[c][0]
            self._set_formants(a, b, self._cons_formants(c, vowel))
            # inside a cluster ("-nds", "-ts") a voiced stop keeps voicing
            self.AV[self._span(a, b)] = (0.25 if cluster else 0.06) if voiced else 0.0
            self.AH[self._span(a, b)] = 0.0
            self._burst(b - 0.002, c, 0.35 if cluster else 0.55)
        elif c in FRIC:
            voiced, _d, fc, bw, amp = FRIC[c]
            self._set_formants(a, b, self._cons_formants(c, vowel))
            self.AV[self._span(a, b)] = 0.3 if voiced else 0.0
            self.AH[self._span(a, b)] = 0.0
            self.fric.append((a, b, fc, bw, amp * 0.85, min(0.012, d * 0.3), min(0.025, d * 0.4)))
        elif c in AFFRIC:
            st, fr = AFFRIC[c]
            m = a + d * 0.5
            self._set_formants(a, b, self._cons_formants(fr, vowel))
            self.AV[self._span(a, b)] = 0.0
            self._burst(m, st, 0.5)
            vf, _d, fc, bw, amp = FRIC[fr]
            self.fric.append((m, b, fc, bw, amp, 0.002, min(0.02, (b - m) * 0.4)))
        elif c in SONORANT_F:
            self._set_formants(a, b, self._cons_formants(c, vowel))
            av = SONORANT_AV[c]
            self.AV[self._span(a, b)] = av
            self.AH[self._span(a, b)] = 0.0
            if c in NASAL_ZERO:
                self.NZ[self._span(a, b)] = NASAL_ZERO[c] * self.fs
            if final:
                r = min(0.035, d * 0.5)
                self._ramp(self.AV, b - r, b, av, 0.0)
        elif c in ("h", "x"):
            self.AV[self._span(a, b)] = 0.0
            self._ramp(self.AH, a, b, 0.3, 0.0)

    # ------------------------------------------------------- planning
    def plan(self, units: list[Unit]):
        """Decide where onset consonants start and where each unit stops."""
        prev = None
        for u in units:
            u.eff_off = u.t_off
            if u.ghost:
                u.eff_off = min(u.t_off, u.t_on + max(0.05, min(0.09, 0.6 * (u.t_off - u.t_on))))
            dur_on = sum(self.cons_dur(c) for c in u.onset)
            start = u.t_on - dur_on
            scale = 1.0
            if prev is not None and prev.eff_off > start - 0.004:
                # the previous syllable is still sounding: the consonant
                # takes time from the end of it, but leaves it at least half
                last = prev.notes[-1]
                limit = max(last.t0 + 0.5 * (last.t1 - last.t0), prev.t_on + 0.03)
                if start < limit and dur_on > 0:
                    start = min(u.t_on - 0.35 * dur_on, max(start, limit))
                    scale = (u.t_on - start) / dur_on
                prev.eff_off = min(prev.eff_off, start)
                prev.joined_next = True
                prev.next_vowel_initial = not u.onset
            u.c_start = max(0.0, start)
            u.c_scale = scale
            prev = u

    # -------------------------------------------------------- writing
    def sing(self, units: list[Unit]):
        self.plan(units)
        for k, u in enumerate(units):
            self._write_unit(u, units[k + 1] if k + 1 < len(units) else None)

    def _write_unit(self, u: Unit, nxt: Unit | None):
        vowels = [p for p in u.nucleus if p in VOWEL_F]
        nuc_son = [p for p in u.nucleus if p in SONORANT_F]
        main = vowels[0] if vowels else None
        glide = vowels[1] if len(vowels) > 1 else None
        hum = nuc_son[0] if (not vowels and nuc_son) else None
        ref_vowel = main or "@"

        # onset consonants
        t = u.c_start
        for c in u.onset:
            d = self.cons_dur(c) * u.c_scale
            self._onset_cons(c, t, t + d, ref_vowel)
            t += d
        t_on = max(u.t_on, t)
        off = u.eff_off
        span = max(0.02, off - t_on)

        # coda layout
        coda = [c for c in u.coda if c != "*ghost*"]
        want = [self._coda_len(c, span, cluster=j + 1 < len(coda)) for j, c in enumerate(coda)]
        total = sum(want)
        cap = 0.45 * span
        if total > cap and total > 0:
            want = [w * cap / total for w in want]
            total = cap
        t_coda = off - total

        # pitch track
        first = True
        for j, note in enumerate(u.notes):
            a = u.c_start if first else note.t0
            b = note.t1 if j + 1 < len(u.notes) else max(off, note.t0 + 0.01)
            sl = self._span(a, b)
            if note.gliss and j + 1 < len(u.notes):
                g0 = note.t0 + 0.4 * (note.t1 - note.t0)
                self.midi[self._span(a, g0)] = note.midi
                self._ramp(self.midi, g0, b, note.midi, u.notes[j + 1].midi)
            else:
                self.midi[sl] = note.midi
            # vibrato on held notes
            nl = note.t1 - note.t0
            if nl > 0.45 and self.vibrato_cents > 0:
                d0 = note.t0 + 0.25
                d1 = min(note.t1, note.t0 + 0.7)
                self._ramp(self.vib, d0, d1, 0.0, self.vibrato_cents)
                self.vib[self._span(d1, note.t1)] = self.vibrato_cents
            first = False
        if not u.legato_in and (u.c_start >= t_on - 0.001 or not u.onset):
            # small scoop into an attack after silence
            self._ramp(self.cents, t_on, t_on + 0.07, -35.0, 0.0)

        # nucleus, note by note (a sonorant override changes the sound)
        cur = hum or main
        for j, note in enumerate(u.notes):
            a = t_on if j == 0 else note.t0
            b = min(note.t1 if j + 1 < len(u.notes) else t_coda, t_coda)
            if b <= a:
                continue
            if note.nucleus:
                cur = note.nucleus
            if cur in VOWEL_F:
                self._set_formants(a, b, self._vowel(cur))
                self.AV[self._span(a, b)] = 1.0
                self.AH[self._span(a, b)] = 0.0
                self.NZ[self._span(a, b)] = 270.0
            else:
                self._set_formants(a, b, self._cons_formants(cur, ref_vowel))
                self.AV[self._span(a, b)] = SONORANT_AV.get(cur, 0.6)
                self.AH[self._span(a, b)] = 0.0
                if cur in NASAL_ZERO:
                    self.NZ[self._span(a, b)] = NASAL_ZERO[cur] * self.fs
        # diphthong off-glide in the last stretch of the nucleus
        if glide and cur == main:
            g = max(t_on + 0.04, t_coda - min(0.16, 0.4 * (t_coda - t_on)))
            if g < t_coda:
                self._set_formants(g, t_coda, self._vowel(glide))

        # coda
        t = t_coda
        for j, (c, w) in enumerate(zip(coda, want)):
            self._coda_cons(c, t, t + w, cur if cur in VOWEL_F else ref_vowel,
                            final=(not u.joined_next) and j == len(coda) - 1,
                            cluster=j + 1 < len(coda))
            t += w

        # release at the cutoff
        if not u.joined_next and (not coda or coda[-1] in SONORANT_F):
            r = min(0.03, 0.3 * span)
            if not coda:
                lvl = self.AV[self._i(off - r)]
                self._ramp(self.AV, off - r, off, lvl, 0.0)
            self.AV[self._span(off, off + 0.002)] = 0.0
        if u.joined_next and u.next_vowel_initial and nxt is not None and not nxt.legato_in:
            self.dips.append((off, 0.045, 0.55))

        # accents
        if u.accent:
            self._ramp(self.accent_db, t_on, t_on + 0.15, u.accent, 0.0)

    def _coda_len(self, c, span, cluster=False):
        if c in SONORANT_DUR and c in NASAL_ZERO:
            return float(np.clip(0.25 * span, 0.06, 0.30))
        if c in SONORANT_DUR:
            return float(np.clip(0.2 * span, 0.05, 0.20))
        if c in STOP:
            return 0.028 if cluster else 0.045
        if c in FRIC:
            return FRIC[c][1] * 1.1
        if c in AFFRIC:
            return 0.09
        return 0.05

    # -------------------------------------------------------- rendering
    def render(self, gain_db: np.ndarray) -> np.ndarray:
        n = self.n
        N = int(self.dur * SR) + BLOCK
        # --- smooth control tracks
        for k in range(5):
            self.F[k] = _hann_smooth(self.F[k], 18)
            self.B[k] = _hann_smooth(self.B[k], 10)
        self.NZ = _hann_smooth(self.NZ, 14)
        AV = _hann_smooth(self.AV, 5)
        AH = _hann_smooth(self.AH, 5)
        for tc, w, depth in self.dips:
            i0, i1 = self._i(tc - w / 2), self._i(tc + w / 2)
            if i1 > i0:
                AV[i0:i1] *= 1.0 - depth * np.hanning(i1 - i0)

        midi = self.midi.copy()
        idx = np.arange(n)
        good = ~np.isnan(midi)
        if not good.any():
            return np.zeros(N, dtype=np.float32)
        midi = np.interp(idx, idx[good], midi[good])
        midi = _hann_smooth(midi, 36)              # legato portamento
        # sopranos open F1 (and F2) above the pitch on high notes
        f0_ctrl = 440.0 * 2 ** ((midi - 69) / 12)
        self.F[0] = np.maximum(self.F[0], f0_ctrl * 1.08)
        self.F[1] = np.maximum(self.F[1], self.F[0] + 250)
        self.F[2] = np.maximum(self.F[2], self.F[1] + 300)
        AV = AV * self._level_comp(f0_ctrl, AV)

        # --- sample-rate tracks
        ts = np.arange(N) / SR
        fpos = ts * CR
        midi_s = np.interp(fpos, idx, midi)
        vib_s = np.interp(fpos, idx, self.vib)
        cents_s = np.interp(fpos, idx, self.cents)
        rate = self.vib_rate * (1 + 0.04 * np.sin(2 * np.pi * 0.23 * ts + self.rng.uniform(0, 6)))
        vphase = np.cumsum(2 * np.pi * rate / SR)
        drift = self._lf_noise(N, 3.0) * 6.0          # slow pitch wander, cents
        jitter = self._lf_noise(N, 40.0) * 2.0
        cents = vib_s * np.sin(vphase) + cents_s + drift + jitter
        f0 = 440.0 * 2 ** ((midi_s - 69 + cents / 100.0) / 12)
        phase = np.cumsum(f0 / SR) % 1.0

        AV_s = np.interp(fpos, idx, AV)
        AH_s = np.interp(fpos, idx, AH)
        lvl = np.interp(fpos, idx, gain_db + self.accent_db)
        # voice quality follows loudness: soft = breathy, loud = pressed
        q = np.clip((lvl + 22) / 26, 0, 1)
        glot = self._glottal(phase, f0, q)
        shimmer = 1 + 0.04 * self._lf_noise(N, 30.0)
        noise = self.rng.standard_normal(N)
        open_mask = (phase < 0.6).astype(float) * 0.7 + 0.3
        breath = noise * open_mask * (self.breathiness * (1.4 - 0.8 * q))
        x = AV_s * (glot * shimmer + breath) + AH_s * noise * 2.0

        y = self._cascade(x, AV_s, AH_s)
        y = sosfilt(_PRESENCE, y)
        y += self._frication(N)
        y *= 10 ** (lvl / 20.0)
        return y.astype(np.float32)

    def _level_comp(self, f0, AV, hop=10):
        """Per-frame gain that cancels the accident of a harmonic landing on
        (or between) formant peaks, so a vowel's loudness doesn't jump around
        with pitch.  Natural differences between vowels are kept."""
        amps = _glottal_tables()[("amps", 1)]
        T = 1.0 / SR
        fr = np.arange(0, self.n, hop)
        on = AV[fr] > 0.02
        comp = np.ones(len(fr))
        if not on.any():
            return np.ones(self.n)
        sel = fr[on]
        F = self.F[:, sel]
        B = self.B[:, sel]
        NZ = self.NZ[sel]
        f0s = f0[sel]

        def mag2(freqs):          # freqs: (frames, m)
            w = 2 * np.pi * freqs * T
            z1 = np.exp(-1j * w)
            z2 = z1 * z1
            out = np.ones(freqs.shape)
            def res(f, bw):
                C = -np.exp(-2 * np.pi * bw * T)
                Bc = 2 * np.exp(-np.pi * bw * T) * np.cos(2 * np.pi * f * T)
                A = 1 - Bc - C
                return A[:, None], Bc[:, None], C[:, None]
            secs = [(np.full(len(sel), 270.0), np.full(len(sel), 100.0), False),
                    (NZ, np.full(len(sel), 100.0), True)]
            secs += [(F[k], B[k], False) for k in range(5)]
            for f, bw, zero in secs:
                A, Bc, C = res(f, bw)
                d = np.abs(1 - Bc * z1 - C * z2) ** 2
                out = out * (d / A ** 2 if zero else A ** 2 / d)
            return out

        K = 64
        k = np.arange(1, K + 1)[None, :]
        fh = f0s[:, None] * k
        valid = fh < 5000
        ah = amps[np.minimum(k - 1, len(amps) - 1)] ** 2
        Ph = np.sum(np.where(valid, ah * mag2(np.minimum(fh, 5000)), 0), axis=1)
        grid = np.linspace(40, 5000, 160)[None, :]
        kk = grid / f0s[:, None]
        ag = np.interp(kk, np.arange(1, len(amps) + 1), amps ** 2)
        Pc = np.sum(ag * mag2(np.broadcast_to(grid, kk.shape)), axis=1) * (grid[0, 1] - grid[0, 0]) / f0s
        c = np.sqrt(Pc / np.maximum(Ph, 1e-12))
        comp[on] = np.clip(c, 0.5, 2.0)
        comp = np.interp(np.arange(self.n), fr, comp)
        return _hann_smooth(comp, 40)

    def _lf_noise(self, N, hz):
        k = max(2, int(N / SR * hz) + 2)
        pts = self.rng.standard_normal(k)
        return np.interp(np.linspace(0, k - 1, N), np.arange(k), pts)

    def _glottal(self, phase, f0, q):
        tabs = _glottal_tables()
        N = len(phase)
        out = np.zeros(N, dtype=np.float64)
        pos = phase * _TABLE_N
        i0 = pos.astype(np.int64) % _TABLE_N
        frac = pos - np.floor(pos)
        i1 = (i0 + 1) % _TABLE_N
        step = 4096
        for s in range(0, N, step):
            e = min(N, s + step)
            fmax = float(np.max(f0[s:e]))
            hmax = int(0.45 * SR / max(fmax, 30.0))
            h = 1
            for cand in (2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
                if cand <= hmax:
                    h = cand
                    break
            qq = q[s:e]
            vals = []
            for qi in range(3):
                t = tabs[qi][h]
                vals.append(t[i0[s:e]] * (1 - frac[s:e]) + t[i1[s:e]] * frac[s:e])
            w_soft = np.clip(1 - 2 * qq, 0, 1)
            w_loud = np.clip(2 * qq - 1, 0, 1)
            w_mid = 1 - w_soft - w_loud
            out[s:e] = w_soft * vals[0] + w_mid * vals[1] + w_loud * vals[2]
        return out

    def _cascade(self, x, AV_s, AH_s):
        N = len(x)
        nb = N // BLOCK
        T = 1.0 / SR
        centers = (np.arange(nb) * BLOCK + BLOCK / 2) / SR * CR
        idx = np.arange(self.n)
        Fb = np.stack([np.interp(centers, idx, self.F[k]) for k in range(5)])
        Bb = np.stack([np.interp(centers, idx, self.B[k]) for k in range(5)])
        NZb = np.interp(centers, idx, self.NZ)
        Fb = np.minimum(Fb, 0.45 * SR)

        def reson(f, bw):
            C = -np.exp(-2 * np.pi * bw * T)
            Bc = 2 * np.exp(-np.pi * bw * T) * np.cos(2 * np.pi * f * T)
            A = 1 - Bc - C
            return A, Bc, C

        nsec = 7
        sos = np.zeros((nb, nsec, 6))
        A, Bc, C = reson(np.full(nb, 270.0), np.full(nb, 100.0))      # nasal pole
        sos[:, 0] = np.stack([A, 0 * A, 0 * A, 1 + 0 * A, -Bc, -C], axis=1)
        A, Bc, C = reson(NZb, np.full(nb, 100.0))                     # nasal zero
        sos[:, 1] = np.stack([1 / A, -Bc / A, -C / A, 1 + 0 * A, 0 * A, 0 * A], axis=1)
        for k in range(5):
            A, Bc, C = reson(Fb[k], Bb[k])
            sos[:, 2 + k] = np.stack([A, 0 * A, 0 * A, 1 + 0 * A, -Bc, -C], axis=1)

        active = np.maximum.reduceat(np.abs(AV_s[:nb * BLOCK]) + np.abs(AH_s[:nb * BLOCK]),
                                     np.arange(0, nb * BLOCK, BLOCK)) > 1e-5
        # keep running a little after sound stops so resonances ring out
        ring = np.convolve(active.astype(float), np.ones(40), mode="full")[:nb] > 0
        y = np.zeros(N)
        if _sosfilt_fast is not None:
            zi = np.zeros((1, nsec, 2))
            sos = np.ascontiguousarray(sos)
            for b in range(nb):
                if not ring[b]:
                    zi[:] = 0.0
                    continue
                s = b * BLOCK
                blk = x[s:s + BLOCK][None, :].copy()
                _sosfilt_fast(sos[b], blk, zi)
                y[s:s + BLOCK] = blk[0]
            return y
        zi = np.zeros((nsec, 2))
        for b in range(nb):
            if not ring[b]:
                zi[:] = 0.0
                continue
            s = b * BLOCK
            y[s:s + BLOCK], zi = sosfilt(sos[b], x[s:s + BLOCK], zi=zi)
        return y

    def _frication(self, N):
        out = np.zeros(N)
        for (t0, t1, fc, bw, amp, att, rel) in self.fric:
            a = int(t0 * SR)
            b = int(t1 * SR)
            if b <= a or a >= N:
                continue
            b = min(b, N)
            L = b - a
            lo = max(100.0, fc - bw / 2)
            hi = min(0.45 * SR, fc + bw / 2)
            if hi <= lo * 1.05:
                continue
            sos = butter(2, [lo, hi], btype="band", fs=SR, output="sos")
            pad = 256
            nz = self.rng.standard_normal(L + pad)
            nz = sosfilt(sos, nz)[pad:]
            env = np.ones(L)
            na = max(1, int(att * SR))
            nr = max(1, int(rel * SR))
            if na < L:
                env[:na] = np.linspace(0, 1, na)
            if nr < L:
                env[-nr:] *= np.linspace(1, 0, nr) ** 1.5
            # normalize the band-passed noise to unit RMS, then scale
            rms = np.sqrt(np.mean(nz ** 2)) + 1e-9
            out[a:b] += nz / rms * env * amp * FRIC_GAIN
        return out


# gentle high shelf (+5 dB above ~3 kHz) for presence
def _shelf():
    from scipy.signal import tf2sos
    A = 10 ** (5 / 40)
    w0 = 2 * np.pi * 3000 / SR
    alpha = np.sin(w0) / 2 * np.sqrt(2)
    cw = np.cos(w0)
    b = [A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * alpha),
         -2 * A * ((A - 1) + (A + 1) * cw),
         A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * alpha)]
    a = [(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * alpha,
         2 * ((A - 1) - (A + 1) * cw),
         (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * alpha]
    return tf2sos(b, a)


_PRESENCE = _shelf()

# Overall level of noise sources relative to the voiced cascade output.
# (Set by calibrate.py so that 's' sits ~10 dB under an 'ah'.)
FRIC_GAIN = 2.6
