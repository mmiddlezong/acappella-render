"""A small Klatt-style formant singing synthesizer.

Each singer is rendered from control tracks sampled every millisecond:
pitch, voicing amplitude, aspiration amplitude, five formants with
bandwidths, and a nasal pole/zero pair.  A band-limited glottal pulse plus
noise goes through a cascade of resonators; fricatives, stop bursts and
breaths are filtered noise added on top.

The tracks are shaped the way a singer moves rather than switched on and
off:
  * attacks rise smoothly: a soft, slightly breathy onset out of silence, a
    quick but rounded rise out of a consonant, and the tone brightens as the
    voice opens up;
  * pitch travels between notes like a damped spring (a little lag, a little
    overshoot), scoops gently into phrase starts and sags at releases;
    vibrato fades in after the onset and never jumps;
  * formants move quickly into a consonant and more slowly out of it into
    the vowel (coarticulation); diphthongs glide;
  * small human variations: timing and tuning scatter, breaths before
    phrases.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

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
    "7": (600, 1050, 2500),      # Korean eo: open-mid back unrounded
    "A": (680, 1020, 2400),      # the round, dark "ah" of a cappella "dah"/"bah"
}
VOWEL_B = (70.0, 90.0, 130.0, 170.0, 220.0)
# How loud each vowel comes out of the vocal tract relative to "a" (dB).
# Singers even this out; half of the difference is made up.
VOWEL_LEVEL_DB = {
    "a": 0.0, "ae": -1.9, "E": -3.1, "e": -3.5, "i": -5.3, "I": -4.0, "o": -0.5,
    "O": 0.6, "u": -2.5, "U": -2.3, "V": -1.4, "@": -2.9, "M": -4.0, "1": -4.0,
    "3": -2.4, "y": -5.0, "7": -0.8, "A": -0.6,
}
VOWEL_GAIN = {v: 10 ** (-0.5 * db / 20.0) for v, db in VOWEL_LEVEL_DB.items()}

PLACE = {}
for c in "b p m f v w p' ph p_}".split():
    PLACE[c] = "lab"
for c in "T D".split():
    PLACE[c] = "dent"
for c in "d t n s z l ts dz R r t' th t_} s'".split():
    PLACE[c] = "alv"
for c in "S Z tS dZ C j S' tS' tSh".split():
    PLACE[c] = "post"
for c in "k g N x k' kh k_}".split():
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
    # Korean tense (fortis): longer closure, crisp burst, almost no aspiration
    "p'": (False, 0.060, 1200, 2000, 0.25, 0.006, 0.008),
    "t'": (False, 0.055, 4500, 3500, 0.40, 0.008, 0.008),
    "k'": (False, 0.060, 2200, 1500, 0.40, 0.012, 0.010),
    # Korean aspirated: long, strong aspiration
    "ph": (False, 0.045, 1200, 2000, 0.28, 0.008, 0.065),
    "th": (False, 0.040, 4500, 3500, 0.42, 0.010, 0.065),
    "kh": (False, 0.045, 2200, 1500, 0.42, 0.014, 0.070),
}
# Korean final stops: the closure is held, never released
UNRELEASED = {"p_}": "p", "t_}": "t", "k_}": "k"}
TENSE = {"p'", "t'", "k'", "s'", "S'", "tS'"}
ASPIRATED = {"ph", "th", "kh", "tSh"}
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
    "s'": (False, 0.100, 6800, 4500, 0.34),     # Korean tense s
    "S'": (False, 0.100, 3400, 2600, 0.38),
}
AFFRIC = {"tS": ("t", "S"), "dZ": ("d", "Z"), "ts": ("t", "s"), "dz": ("d", "z"),
          "tS'": ("t'", "S"), "tSh": ("th", "S")}
SONORANT_DUR = {"m": 0.065, "n": 0.060, "N": 0.065, "l": 0.055, "r": 0.065, "R": 0.022,
                "w": 0.060, "j": 0.055}
SONORANT_AV = {"m": 0.50, "n": 0.50, "N": 0.50, "l": 0.70, "r": 0.75, "R": 0.55,
               "w": 0.75, "j": 0.80}
# a nasal before or after a vowel is a murmur, quieter than a sung hum
MURMUR_AV = {"m": 0.30, "n": 0.30, "N": 0.32}
# a hummed syllable ("mm", "dn", the n of "fu-n"); keeps hums about as far
# under an open vowel as they have always been here
HUM_AV = 0.66
VOICED_CONS = set(SONORANT_F) | {"b", "d", "g", "z", "Z", "v", "D", "dZ", "dz"}

# Breathy noise at the start of a vowel after p/t/k (start, end level)
ASP_LEVEL = (0.24, 0.10)
# Stop bursts are short; sung, they sit well under the vowel
BURST_GAIN = 0.16

# Formant smoothing half-widths in ms: how quickly the vocal tract moves
W_VOWEL = 16      # out of a consonant into a vowel (~35 ms transition)
W_CONS = 4        # into a closure / fricative
W_NASAL = 6
W_LIQUID = 7
W_GLIDE = 12      # w, j, r are slow gestures

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
    breath_after: bool = False     # a breath mark is written after this note


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
    word: bool = False             # a lyric word (not a vocable like doo/dm)
    perc: str = ""                 # vocal percussion hit (snare/kick/hat), not sung
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
# Voice quality from soft/breathy (q=0) to pressed (q=1): open quotient and
# extra spectral tilt.  Neighbouring tables are cross-faded.
_Q_LEVELS = 7
_Q_OQ = (0.72, 0.45)
_Q_TILT = (0.38, -0.18)
_CLOSE_PHASE = 0.72        # every pulse closes (its main excitation) here


def _glottal_tables():
    """Band-limited glottal-flow-derivative tables for _Q_LEVELS voice
    qualities x several harmonic limits.  All pulses are aligned at the
    moment of glottal closure, so cross-fading two qualities gives an
    in-between pulse instead of two half-size excitations that partly
    cancel (which made loudness jump around as the quality changed)."""
    global _TABLES
    if _TABLES is not None:
        return _TABLES
    hi = 1 << 15
    t = np.arange(hi) / hi
    out = {}
    for qi in range(_Q_LEVELS):
        x = qi / (_Q_LEVELS - 1)
        oq = _Q_OQ[0] + (_Q_OQ[1] - _Q_OQ[0]) * x
        tilt = _Q_TILT[0] + (_Q_TILT[1] - _Q_TILT[0]) * x
        t0 = _CLOSE_PHASE - oq                      # glottis opens here
        tt = (t - t0) % 1.0
        u = np.where(tt < oq, (tt / oq) ** 2 - (tt / oq) ** 3, 0.0)
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


def _damping(z: float, interval: float) -> float:
    """Big leaps overshoot proportionally less than steps do."""
    return float(min(0.9, z + 0.02 * max(0.0, abs(interval) - 2.0)))


def _var_smooth(x: np.ndarray, W: np.ndarray, passes: int = 2) -> np.ndarray:
    """Centered moving average whose half-width W[i] (frames) changes along
    the track.  Two passes give a triangular kernel: wide windows in vowels
    make slow, rounded transitions, narrow ones in consonants quick ones."""
    n = len(x)
    W = np.asarray(W, dtype=np.int64)
    pad = int(W.max()) + 1
    idx = np.arange(n) + pad
    lo = idx - W
    hi = idx + W + 1
    cnt = (hi - lo).astype(float)
    for _ in range(passes):
        xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
        C = np.concatenate([[0.0], np.cumsum(xp)])
        x = (C[hi] - C[lo]) / cnt
    return x


# ------------------------------------------------------------------ singer
class Singer:
    def __init__(self, kind: str, duration: float, seed: int = 0,
                 vibrato_cents: float = 15.0, breathiness: float = 0.06,
                 humanize: float = 1.0, breaths: float = 1.0, attack: float = 1.0,
                 scoop_cents: float = 25.0, formant_shift: float = 1.0,
                 detune_cents: float = 0.0):
        self.kind = kind if kind in PROFILES else "baritone"
        fs, f4, f5, self.vib_rate = PROFILES[self.kind]
        self.fs, self.f4, self.f5 = fs * formant_shift, f4 * formant_shift, f5 * formant_shift
        self.dur = duration
        self.n = int(duration * CR) + 2
        self.rng = np.random.default_rng(seed)
        self.hrng = np.random.default_rng(seed + 104729)    # humanization only
        self.vibrato_cents = vibrato_cents
        self.breathiness = breathiness
        self.humanize = humanize
        self.breaths = breaths
        self.attack = max(0.2, attack)
        self.scoop_cents = scoop_cents
        self.detune = detune_cents
        n = self.n
        neutral = self._vowel("@")
        self.F = np.tile(np.array(neutral[0], dtype=float)[:, None], (1, n))
        self.B = np.tile(np.array(neutral[1], dtype=float)[:, None], (1, n))
        self.SW = np.full(n, W_VOWEL, dtype=np.int64)    # formant smoothing width
        self.AV = np.zeros(n)
        self.AH = np.zeros(n)
        self.NZ = np.full(n, 270.0)      # nasal zero; 270 == pole -> no nasality
        self.vib = np.zeros(n)          # vibrato depth in cents
        self.vrate = np.full(n, self.vib_rate)
        self.cents = np.zeros(n)        # release sags
        self.accent_db = np.zeros(n)
        self.qoff = np.zeros(n)         # voice-quality push (tense onsets)
        self.bloom = np.ones(n)         # 0 = voice just opening (darker), 1 = open
        self.fric: list[tuple] = []     # (t0, t1, fc, bw, amp, attack, release)
        self.dips: list[tuple] = []     # (t_center, width, depth)
        self.breath_ev: list[tuple] = []  # (t0, t1, amp, brightness)
        self.pplan: list[tuple] = []    # pitch targets (t, midi, T, zeta, reset, scoop)
        self.gain = None
        self.units: list[Unit] = []
        self.mid_pitch = 60.0
        self._reset_idx = None
        self.air_end = 1.0
        self.air_warnings: list[tuple] = []     # (phrase start, runs out at, seconds short)

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
        c = UNRELEASED.get(c, c)
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

    def _rcos(self, arr, a, b, v0, v1):
        """Raised-cosine move from v0 to v1 over [a, b): starts and ends flat."""
        sl = self._span(a, b)
        m = sl.stop - sl.start
        if m > 1:
            x = np.linspace(0.0, 1.0, m)
            arr[sl] = v0 + (v1 - v0) * (0.5 - 0.5 * np.cos(np.pi * x))
        else:
            arr[sl] = v1

    def _gain_at(self, t: float) -> float:
        if self.gain is None or len(self.gain) == 0:
            return -5.0
        return float(self.gain[int(np.clip(round(t * CR), 0, len(self.gain) - 1))])

    @staticmethod
    def cons_dur(c: str) -> float:
        c = UNRELEASED.get(c, c)
        if c in STOP:
            v, clo, *_r, asp = STOP[c]
            # a sung b/d/g is a quick tap rather than a full stop
            return clo * (0.9 if v else 1.0) + asp
        if c in FRIC:
            return FRIC[c][1]
        if c in AFFRIC:
            st, fr = AFFRIC[c]
            extra = 0.025 if c in ASPIRATED else 0.0
            return STOP[st][1] * 0.8 + FRIC[fr][1] * 0.6 + extra
        if c in SONORANT_DUR:
            return SONORANT_DUR[c]
        if c in ("h", "x"):
            return 0.060
        return 0.04

    # ------------------------------------------------- consonant writers
    def _burst(self, t, c, amp_scale=1.0):
        voiced, clo, fc, bw, amp, blen, asp = STOP[c]
        fc = fc * (1.1 if self.fs > 1.08 else 1.0)
        g = BURST_GAIN * (0.65 if voiced else 1.0)
        self.fric.append((t, t + blen + 0.004, fc, bw, amp * amp_scale * g, 0.0015, blen))

    def _onset_cons(self, c, a, b, vowel, after_silence=False):
        """Write consonant c occupying [a, b) before a vowel."""
        d = b - a
        c = UNRELEASED.get(c, c)
        if c in STOP:
            voiced, clo, fc, bw, amp, blen, asp = STOP[c]
            frac_clo = clo / (clo + asp) if asp else 1.0
            tb = a + d * frac_clo
            sl = self._span(a, tb)
            self._set_formants(a, tb, self._cons_formants(c, vowel))
            self.SW[sl] = W_CONS
            if voiced:
                # a sung b/d/g between vowels keeps voicing through the closure
                self.AV[sl] = 0.05 if after_silence else 0.2
                self.F[0, sl] = 210 * self.fs
            else:
                self.AV[sl] = 0.0
            self.AH[sl] = 0.0
            self._burst(tb, c)
            if asp:
                self._set_formants(tb, b, self._vowel(vowel))
                self.AV[self._span(tb, b)] = 0.0
                boost = 1.3 if c in ASPIRATED else 1.0
                self._ramp(self.AH, tb, b, ASP_LEVEL[0] * boost, ASP_LEVEL[1] * boost)
        elif c in FRIC:
            voiced, _d, fc, bw, amp = FRIC[c]
            sl = self._span(a, b)
            self._set_formants(a, b, self._cons_formants(c, vowel))
            self.SW[sl] = W_CONS
            self.AV[sl] = (0.2 if after_silence else 0.35) if voiced else 0.0
            self.AH[sl] = 0.0
            self.fric.append((a, b, fc * (1.08 if self.fs > 1.08 else 1.0), bw, amp,
                              min(0.015, d * 0.3), min(0.012, d * 0.3)))
        elif c in AFFRIC:
            st, fr = AFFRIC[c]
            m = a + d * (0.4 if c in ASPIRATED else 0.45)
            voiced = STOP[st][0]
            self._set_formants(a, m, self._cons_formants(fr, vowel))
            self.SW[self._span(a, b)] = W_CONS
            self.AV[self._span(a, m)] = (0.05 if after_silence else 0.12) if voiced else 0.0
            self.AH[self._span(a, m)] = 0.0
            self._burst(m, st, 0.6)
            vf, _d, fc, bw, amp = FRIC[fr]
            self._set_formants(m, b, self._cons_formants(fr, vowel))
            self.AV[self._span(m, b)] = 0.3 if voiced else 0.0
            self.fric.append((m, b, fc, bw, amp * (1.25 if c in ASPIRATED else 1.1),
                              0.002, min(0.012, (b - m) * 0.4)))
        elif c in SONORANT_F:
            sl = self._span(a, b)
            self._set_formants(a, b, self._cons_formants(c, vowel))
            av = MURMUR_AV.get(c, SONORANT_AV[c])
            self.AV[sl] = av
            self.AH[sl] = 0.0
            if after_silence:      # the murmur itself swells in
                self._rcos(self.AV, a, a + min(0.045 * self.attack, 0.7 * d), 0.0, av)
                self._rcos(self.bloom, a, a + min(0.045 * self.attack, 0.7 * d), 0.0, 1.0)
            if c in NASAL_ZERO:
                self.NZ[sl] = NASAL_ZERO[c] * self.fs
                self.SW[sl] = W_NASAL
            else:
                self.SW[sl] = W_GLIDE if c in ("w", "j", "r") else W_LIQUID
        elif c in ("h", "x"):
            sl = self._span(a, b)
            self._set_formants(a, b, self._vowel(vowel))
            if after_silence:
                self.AV[sl] = 0.0
                self._ramp(self.AH, a, b, 0.10, 0.28)
            else:
                # between sung sounds h is breathy voice, not a gap
                self.AV[sl] = 0.30
                self._ramp(self.AH, a, b, 0.22, 0.14)
            if c == "x":
                self.fric.append((a, b, 1600 * self.fs, 1200, 0.06, 0.01, 0.01))

    def _coda_cons(self, c, a, b, vowel, final: bool, cluster: bool = False):
        """Write consonant c occupying [a, b) after a vowel.  `final` means
        the sound stops at b (rather than running into the next syllable)."""
        d = b - a
        sl = self._span(a, b)
        if c in UNRELEASED:
            # held closure: the vowel is cut off by the lips/tongue, no pop
            self._set_formants(a, b, self._cons_formants(c, vowel))
            self.SW[sl] = W_CONS
            self.AV[sl] = 0.0
            self.AH[sl] = 0.0
        elif c in STOP:
            voiced = STOP[c][0]
            self._set_formants(a, b, self._cons_formants(c, vowel))
            self.SW[sl] = W_CONS
            # inside a cluster ("-nds", "-ts") a voiced stop keeps voicing
            self.AV[sl] = (0.25 if cluster else 0.08) if voiced else 0.0
            self.AH[sl] = 0.0
            self._burst(b - 0.002, c, 0.35 if cluster else 0.55)
        elif c in FRIC:
            voiced, _d, fc, bw, amp = FRIC[c]
            self._set_formants(a, b, self._cons_formants(c, vowel))
            self.SW[sl] = W_CONS
            self.AV[sl] = 0.3 if voiced else 0.0
            self.AH[sl] = 0.0
            self.fric.append((a, b, fc, bw, amp * 0.85, min(0.012, d * 0.3), min(0.025, d * 0.4)))
        elif c in AFFRIC:
            st, fr = AFFRIC[c]
            m = a + d * 0.5
            self._set_formants(a, b, self._cons_formants(fr, vowel))
            self.SW[sl] = W_CONS
            self.AV[sl] = 0.0
            self._burst(m, st, 0.5)
            vf, _d, fc, bw, amp = FRIC[fr]
            self.fric.append((m, b, fc, bw, amp, 0.002, min(0.02, (b - m) * 0.4)))
        elif c in SONORANT_F:
            self._set_formants(a, b, self._cons_formants(c, vowel))
            av = SONORANT_AV[c] if c not in NASAL_ZERO else 0.42
            self.AV[sl] = av
            self.AH[sl] = 0.0
            if c in NASAL_ZERO:
                self.NZ[sl] = NASAL_ZERO[c] * self.fs
                self.SW[sl] = W_NASAL
            else:
                self.SW[sl] = W_LIQUID
            if final:
                r = float(np.clip(0.5 * d, 0.03, 0.08))
                self._rcos(self.AV, b - r, b, av, 0.0)
        elif c in ("h", "x"):
            self.AV[sl] = 0.0
            self._ramp(self.AH, a, b, 0.2, 0.0)

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

    def _humanize(self, units: list[Unit]):
        """Nobody sings exactly on the grid or exactly in tune: scatter note
        onsets by a few ms and tuning by a few cents.  Boundaries between
        touching notes move together, so nothing opens a gap."""
        h = self.humanize
        if h <= 0:
            return
        rng = self.hrng
        prev = None
        for u in units:
            n0 = u.notes[0]
            d = float(np.clip(rng.normal(0.0, 0.007 * h), -0.018 * h, 0.018 * h))
            if prev is not None and abs(prev.notes[-1].t1 - n0.t0) < 0.006:
                p = prev.notes[-1]
                lim = 0.25 * min(p.t1 - p.t0, n0.t1 - n0.t0)
                d = float(np.clip(d, -lim, lim))
                p.t1 += d
            else:
                d = float(np.clip(d, -0.25 * (n0.t1 - n0.t0), 0.25 * (n0.t1 - n0.t0)))
            n0.t0 = max(0.0, n0.t0 + d)
            for note in u.notes:
                note.midi += float(np.clip(rng.normal(0.0, 0.03 * h), -0.08, 0.08))
            prev = u

    # -------------------------------------------------------- writing
    def sing(self, units: list[Unit], gain_db: np.ndarray | None = None):
        units = copy.deepcopy([u for u in units if not u.perc])
        self.gain = gain_db
        pitches = [nt.midi for u in units for nt in u.notes]
        self.mid_pitch = float(np.median(pitches)) if pitches else 60.0
        self._humanize(units)
        self.plan(units)
        for k, u in enumerate(units):
            self._write_unit(u, units[k - 1] if k else None,
                             units[k + 1] if k + 1 < len(units) else None)
        self._plan_breaths(units)
        self.units = units

    def _write_unit(self, u: Unit, prev: Unit | None, nxt: Unit | None):
        if not u.word:
            # backing vocables: the round "ah" of a cappella "dah", and a
            # warmer, less pressed tone that blends
            u.nucleus = ["A" if p == "a" else p for p in u.nucleus]
            if u.nucleus and u.nucleus[0] in ("A", "ae", "V", "O", "E", "7"):
                self.qoff[self._span(u.c_start, u.eff_off + 0.05)] -= 0.12
        vowels = [p for p in u.nucleus if p in VOWEL_F]
        nuc_son = [p for p in u.nucleus if p in SONORANT_F]
        main = vowels[0] if vowels else None
        glide = vowels[1] if len(vowels) > 1 else None
        hum = nuc_son[0] if (not vowels and nuc_son) else None
        ref_vowel = main or "@"
        joined_in = prev is not None and prev.joined_next
        qd = float(np.clip((self._gain_at(u.t_on) + 22) / 26, 0, 1))

        # onset consonants
        t = u.c_start
        for k, c in enumerate(u.onset):
            d = self.cons_dur(c) * u.c_scale
            self._onset_cons(c, t, t + d, ref_vowel, after_silence=(k == 0 and not joined_in))
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

        # pitch targets and vibrato
        self._plan_pitch(u, prev, joined_in, t_on, off)

        # nucleus, note by note (a sonorant override changes the sound)
        cur = hum or main
        for j, note in enumerate(u.notes):
            a = t_on if j == 0 else note.t0
            b = min(note.t1 if j + 1 < len(u.notes) else t_coda, t_coda)
            if b <= a:
                continue
            if note.nucleus:
                cur = note.nucleus
            sl = self._span(a, b)
            if cur in VOWEL_F:
                self._set_formants(a, b, self._vowel(cur))
                self.AV[sl] = VOWEL_GAIN.get(cur, 1.0)
                self.AH[sl] = 0.0
                self.NZ[sl] = 270.0
            else:
                self._set_formants(a, b, self._cons_formants(cur, ref_vowel))
                self.AV[sl] = HUM_AV if cur in NASAL_ZERO else SONORANT_AV.get(cur, 0.6)
                self.AH[sl] = 0.0
                if cur in NASAL_ZERO:
                    self.NZ[sl] = NASAL_ZERO[cur] * self.fs
        # diphthong off-glide: a real glide in the last stretch of the nucleus
        if glide and cur == main:
            g = max(t_on + 0.04, t_coda - min(0.18, 0.45 * (t_coda - t_on)))
            if g < t_coda:
                (f0v, _b0), (f1v, _b1) = self._vowel(main), self._vowel(glide)
                for k in range(5):
                    self._rcos(self.F[k], g, t_coda, f0v[k], f1v[k])
                self._rcos(self.AV, g, t_coda, VOWEL_GAIN.get(main, 1.0), VOWEL_GAIN.get(glide, 1.0))

        self._shape_attack(u, prev, joined_in, t_on, t_coda, qd, hum or main, ref_vowel)

        # coda
        t = t_coda
        for j, (c, w) in enumerate(zip(coda, want)):
            self._coda_cons(c, t, t + w, cur if cur in VOWEL_F else ref_vowel,
                            final=(not u.joined_next) and j == len(coda) - 1,
                            cluster=j + 1 < len(coda))
            t += w

        # release at the cutoff
        if not u.joined_next and (not coda or coda[-1] in SONORANT_F):
            gap_after = (nxt.c_start - off) if nxt is not None else 9.0
            if not coda:
                r = min(float(np.clip(0.3 * span, 0.035, 0.09)), 0.5 * span)
                lvl = self.AV[self._i(off - r)]
                self._rcos(self.AV, off - r, off, lvl, 0.0)
                self._rcos(self.bloom, off - r, off, 1.0, 0.5)
                if span > 0.2 and gap_after > 0.08 and vowels:
                    # the air keeps going a moment after the tone stops
                    h = 0.05 * (1.0 - 0.4 * qd)
                    self._rcos(self.AH, off - 0.5 * r, off, 0.0, h)
                    self._rcos(self.AH, off, off + 0.04, h, 0.0)
                if span > 0.3 and gap_after > 0.12 and not u.ghost:
                    fall = 12.0 + 10.0 * (1.0 - qd)       # cents
                    self._rcos(self.cents, off - r, off + 0.03, 0.0, -fall)
                    self.cents[self._span(off + 0.03, off + 0.08)] = -fall
            self.AV[self._span(off, off + 0.002)] = 0.0
        if u.joined_next and u.next_vowel_initial and nxt is not None and not nxt.legato_in:
            self.dips.append((off, 0.05, 0.5))

        # accents: a quick swell into the attack, then back
        if u.accent:
            a0 = u.accent
            self._rcos(self.accent_db, t_on - 0.012, t_on + 0.008, 0.0, a0)
            self._rcos(self.accent_db, t_on + 0.008, t_on + 0.16, a0, 0.0)

    def _shape_attack(self, u, prev, joined_in, t_on, t_end, qd, first_sound, ref_vowel):
        """Replace the switch-on of the nucleus with a sung rise."""
        is_vowel = first_sound in VOWEL_F
        room = t_end - t_on
        if room < 0.012:
            return
        i_on = self._i(t_on)
        target = float(self.AV[min(self.n - 1, i_on + 1)])
        if target <= 0:
            return
        last = u.onset[-1] if u.onset else None
        soft = self.attack
        acc = u.accent > 0.5
        if last is None and joined_in:
            # vowel after vowel inside a phrase: the dip handles it.  After a
            # voiceless coda ("but I", "what a") the voice restarts: shape it
            v0 = float(self.AV[max(0, i_on - 1)])
            if v0 >= 0.6 * target:
                return
            att = min(0.035 * soft * (0.6 if acc else 1.0), 0.5 * room)
            self._rcos(self.AV, t_on, t_on + att, v0, target)
            self._rcos(self.bloom, t_on, t_on + att, v0 / target, 1.0)
            return
        if last is None:
            # out of silence: a soft onset, led by a little air
            att = (0.12 - 0.05 * qd) * soft * (0.4 if acc else 1.0)
            a0 = t_on - 0.3 * att
            if prev is not None:
                a0 = max(a0, prev.eff_off + 0.005)
            # a short note still has to sound: the rise is done by mid-note
            att = max(0.004, min(a0 + att, t_on + 0.5 * room) - a0)
            # the mouth shape and the pitch are ready before the voice starts
            pre = max(a0 - 0.04, (prev.eff_off + 0.005) if prev is not None else 0.0)
            if pre < t_on and first_sound is not None:
                if is_vowel:
                    self._set_formants(pre, t_on, self._vowel(first_sound))
                else:
                    self._set_formants(pre, t_on, self._cons_formants(first_sound, ref_vowel))
                    if first_sound in NASAL_ZERO:
                        self.NZ[self._span(pre, t_on)] = NASAL_ZERO[first_sound] * self.fs
            if self._reset_idx is not None:
                t, m, T, z, r, sc = self.pplan[self._reset_idx]
                self.pplan[self._reset_idx] = (min(t, a0 - 0.003), m, T, z, r, sc)
            self._rcos(self.AV, a0, a0 + att, 0.0, target)
            self._rcos(self.bloom, a0, a0 + att, 0.0, 1.0)
            h = 0.09 * (1.0 - 0.5 * qd) * (0.3 if acc else 1.0) * min(1.5, soft)
            if is_vowel and h > 0.005:
                b0 = a0 - 0.025
                if prev is not None:
                    b0 = max(b0, prev.eff_off + 0.005)
                self._rcos(self.AH, b0, a0 + 0.25 * att, 0.0, h)
                self._rcos(self.AH, a0 + 0.25 * att, a0 + 0.9 * att, h, 0.0)
            return
        # out of a consonant
        base = UNRELEASED.get(last, last)
        if base in STOP:
            att = 0.009 if STOP[base][0] else 0.015    # a stop releases crisply
        elif base in AFFRIC:
            att = 0.012 if STOP[AFFRIC[base][0]][0] else 0.018
        elif base in FRIC or base in ("h", "x"):
            att = 0.035
        elif base in ("w", "j", "r"):
            att = 0.06
        else:
            att = 0.045
        if last in TENSE:
            att = 0.014
        att *= soft * (0.6 if acc else 1.0)
        att = min(att, 0.5 * room)
        v0 = min(target, float(self.AV[max(0, i_on - 1)]))
        self._rcos(self.AV, t_on, t_on + att, v0, target)
        b0 = 0.55 if base in STOP or base in AFFRIC else 0.35 + 0.65 * v0 / target
        self._rcos(self.bloom, t_on, t_on + att, b0, 1.0)
        # aspiration / breathy noise fades under the vowel instead of stopping
        h0 = float(self.AH[max(0, i_on - 1)])
        if h0 > 0:
            self._rcos(self.AH, t_on, t_on + max(0.012, 0.7 * att), h0, 0.0)
        if last in TENSE:
            self._rcos(self.qoff, t_on, t_on + 0.08, 0.3, 0.0)

    def _coda_len(self, c, span, cluster=False):
        c0 = UNRELEASED.get(c, c)
        if c0 in SONORANT_DUR and c0 in NASAL_ZERO:
            return float(np.clip(0.25 * span, 0.06, 0.30))
        if c0 in SONORANT_DUR:
            return float(np.clip(0.2 * span, 0.05, 0.20))
        if c in UNRELEASED:
            return 0.03 if cluster else 0.05
        if c0 in STOP:
            return 0.028 if cluster else 0.045
        if c0 in FRIC:
            return FRIC[c0][1] * 1.1
        if c0 in AFFRIC:
            return 0.09
        return 0.05

    # ----------------------------------------------------------- pitch
    def _plan_pitch(self, u: Unit, prev: Unit | None, joined_in: bool, t_on: float, off: float):
        """Pitch targets: (time, midi, transition time, damping, reset, scoop).
        A reset starts the voice a little flat (a scoop) out of silence."""
        det = self.detune / 100.0
        notes = u.notes
        first = notes[0]
        m0 = first.midi + det
        self._reset_idx = None
        if not joined_in:
            self._reset_idx = len(self.pplan)
            # the pitch is set where the voice first sounds: the first voiced
            # consonant ("w" in "sw", "l" in "pl"), else the vowel
            ts, tc = t_on, u.c_start
            for c in u.onset:
                if c in VOICED_CONS:
                    ts = tc
                    break
                tc += self.cons_dur(c) * u.c_scale
            nl = max(0.02, (first.t1 if len(notes) > 1 else off) - t_on)
            gap = u.c_start - (prev.eff_off if prev is not None else -1.0)
            scoop = self.scoop_cents * float(np.clip(nl / 0.4, 0.35, 1.0))
            if gap < 0.25:
                scoop *= 0.5
            if u.accent > 0.5:
                scoop *= 0.6
            self.pplan.append((ts - 0.002, m0, 0.07, 0.62, True, scoop / 100.0))
        elif u.onset:
            # the move happens mostly during the consonant, so the vowel
            # arrives (nearly) in tune; it may start bending the end of the
            # previous note, but not its first half
            pn = prev.notes[-1]
            T = float(np.clip(0.4 * (t_on - u.c_start), 0.018, 0.035))
            earliest = max(pn.t0 + 0.5 * (pn.t1 - pn.t0), prev.t_on + 0.03)
            ts = max(earliest, t_on - 1.4 * T)
            z = _damping(0.8, m0 - pn.midi)
            self.pplan.append((min(ts, t_on - 0.01), m0, T, z, False, 0.0))
        elif u.legato_in:
            # slurred: the glide is centred on the note change
            pn = prev.notes[-1]
            dl = min(pn.t1 - pn.t0, first.t1 - first.t0)
            T = float(np.clip(0.1 * dl, 0.02, 0.035))
            self.pplan.append((t_on - 0.8 * T, m0, T, _damping(0.75, m0 - pn.midi), False, 0.0))
        else:
            # a vowel re-sung without a slur: a quick move through the
            # little break between the two vowels
            pn = prev.notes[-1]
            T = 0.022
            self.pplan.append((t_on - 1.1 * T, m0, T, _damping(0.8, m0 - pn.midi), False, 0.0))
        for j, note in enumerate(notes):
            if j > 0:
                dl = min(notes[j - 1].t1 - notes[j - 1].t0, note.t1 - note.t0)
                T = float(np.clip(0.1 * dl, 0.02, 0.035))
                z = _damping(0.75, note.midi - notes[j - 1].midi)
                self.pplan.append((note.t0 - 0.8 * T, note.midi + det, T, z, False, 0.0))
            if note.gliss and j + 1 < len(notes):
                g0 = note.t0 + 0.4 * (note.t1 - note.t0)
                g1 = note.t1
                ma, mb = note.midi + det, notes[j + 1].midi + det
                steps = max(2, int((g1 - g0) / 0.01))
                for s in range(1, steps + 1):
                    self.pplan.append((g0 + (g1 - g0) * s / steps - 0.02,
                                       ma + (mb - ma) * s / steps, 0.03, 0.7, False, 0.0))
        self._plan_vibrato(u, t_on, off)

    def _plan_vibrato(self, u: Unit, t_on: float, off: float):
        if self.vibrato_cents <= 0:
            return
        prev_depth = 0.0
        for j, note in enumerate(u.notes):
            a = t_on if j == 0 else note.t0
            b = note.t1 if j + 1 < len(u.notes) else off
            nl = b - a
            if nl < 0.3:
                prev_depth = 0.0
                continue
            depth = self.vibrato_cents * float(self.rng.uniform(0.85, 1.15))
            if prev_depth > 0:          # a held line: the vibrato keeps going
                d0, start = a + 0.06, 0.5 * prev_depth
                d1 = d0 + 0.18
            else:
                d0, start = a + float(np.clip(0.2 + 0.2 * nl, 0.28, 0.45)), 0.0
                d1 = d0 + float(np.clip(0.35 * nl, 0.2, 0.5))
            held = b - d0
            if held < 0.3:
                # not enough time for a couple of full cycles: a lone wobble
                # sounds like a wrong pitch, so stay straight
                prev_depth = 0.0
                continue
            depth *= float(np.clip((held - 0.3) / 0.4 + 0.4, 0.4, 1.0))
            d1 = min(d1, b)
            self.vib[self._span(a, d0)] = start
            self._rcos(self.vib, d0, d1, start, depth)
            self.vib[self._span(d1, b)] = depth
            rate = self.vib_rate * float(self.rng.uniform(0.95, 1.05))
            self.vrate[self._span(a, b)] = rate
            if nl > 0.8:      # singers' vibrato speeds up a touch toward the end
                self._ramp(self.vrate, b - 0.3 * nl, b, rate, rate * 1.06)
            prev_depth = depth

    def _pitch_contour(self):
        """Pitch at control rate: each target is approached like a damped
        spring, so notes arrive with a little lag and overshoot."""
        # targets are planned in musical order; a target never jumps ahead of
        # the one before it (a very short first note could otherwise have its
        # successor's target land before its own reset)
        ev, last_t = [], -1e9
        for e in self.pplan:
            last_t = max(e[0], last_t)
            ev.append((last_t,) + tuple(e[1:]))
        if not ev:
            return None
        n = self.n
        out = np.empty(n)
        dt = 1.0 / CR
        tgt, w, z = ev[0][1], 1.8 / 0.07, 0.62
        y, v = tgt, 0.0
        i_cur = 0
        for (t, m, T, zeta, reset, scoop) in ev:
            i = int(np.clip(round(t * CR), 0, n))
            if i > i_cur:
                k = i - i_cur
                tt = np.arange(k + 1) * dt
                e0 = y - tgt
                wd = w * np.sqrt(1.0 - z * z)
                A, B = e0, (v + z * w * e0) / wd
                ex = np.exp(-z * w * tt)
                c, s = np.cos(wd * tt), np.sin(wd * tt)
                e = ex * (A * c + B * s)
                out[i_cur:i] = tgt + e[:k]
                y = tgt + e[k]
                v = ex[k] * (-z * w * (A * c[k] + B * s[k]) + wd * (B * c[k] - A * s[k]))
                i_cur = i
            if reset:
                y, v = m - scoop, 0.0
            tgt, w, z = m, 1.8 / max(T, 0.01), min(0.95, zeta)
        if i_cur < n:
            k = n - i_cur
            tt = np.arange(k) * dt
            e0 = y - tgt
            wd = w * np.sqrt(1.0 - z * z)
            A, B = e0, (v + z * w * e0) / wd
            out[i_cur:] = tgt + np.exp(-z * w * tt) * (A * np.cos(wd * tt) + B * np.sin(wd * tt))
        return out

    # --------------------------------------------------------- breaths
    def _air_rate(self, t: float, u: Unit | None = None) -> float:
        """Fraction of a full breath used per second of singing: a full
        breath lasts about 20 s at mf, 14 s at f, 10 s at ff, 30 s at pp.
        A hum uses less air than an open vowel."""
        r = 0.05 * 10 ** (0.6 * (self._gain_at(t) + 5.0) / 20.0)
        if u is not None and not any(p in VOWEL_F for p in u.nucleus):
            r *= 0.6
        return r

    def _plan_breaths(self, units: list[Unit]):
        """Breaths from a running model of the singer's air.

        Air (0..1 of a full breath) drains while singing, faster when loud,
        and is refilled in the gaps.  Before each phrase the singer takes in
        what the stretch ahead needs plus a reserve (and tops up more when
        there is time).  The breath's size and the time available set how
        it sounds: a big refill in a short gap is a quick, loud, hissy gasp;
        a top-up in a long rest is slow and quiet.  Short gaps get a quick
        catch breath when the singer is getting low; a breath mark always
        gets a breath.  Stretches that still run out of air are recorded in
        air_warnings (they don't change the sound)."""
        self.air_warnings = []
        if not units:
            return
        # phrases: runs of joined units (a breath mark always ends one)
        phrases, cur = [], [units[0]]
        for u in units[1:]:
            if cur[-1].joined_next and not cur[-1].notes[-1].breath_after:
                cur.append(u)
            else:
                phrases.append(cur)
                cur = [u]
        phrases.append(cur)

        def need(ph):
            return sum(self._air_rate(u.c_start, u) * max(0.0, u.eff_off - u.c_start) for u in ph)

        air = 0.6                         # a normal breath in before the piece
        prev_end = 0.0
        stretch = None                    # [start, ran out at, seconds short] since the last refill
        for k, ph in enumerate(phrases):
            start = ph[0].c_start
            gap = start - prev_end
            marked = k > 0 and phrases[k - 1][-1].notes[-1].breath_after
            ahead = need(ph)
            for nxt in phrases[k + 1:]:
                if nxt[0].c_start - ph[-1].eff_off >= 0.5:
                    break
                ahead += need(nxt)
            want = max(0.0, min(1.0, ahead + 0.12) - air)
            if gap >= 0.7:
                want = max(want, 0.5 * (1.0 - air))      # a long rest: top up anyway
            elif gap < 0.3 and not marked and air > 0.45:
                want = 0.0                               # plenty left: sing on
            want = min(want, 1.0 - air)
            room = gap - 0.13
            if gap < 0.3:
                # a quick sniff between notes; tighter the lower the singer is
                room = gap - (0.04 if air < 0.35 else 0.08)
            if k == 0:
                room = min(room, 0.5)
            if marked:
                # a written breath: the singer makes the time for a real one
                room = max(room, gap - 0.03, 0.3)
            taken = 0.0
            if want >= 0.04 and room >= 0.05:
                comfy = 0.25 + 0.6 * want                # an unhurried breath this size
                dur = float(min(room, comfy, 0.9))
                # how much air fits in the time: about 1.6 breaths/s at most
                if marked:
                    taken = min(want, 2.5 * dur)
                else:
                    taken = min(want, 1.6 * dur, want * min(1.0, dur / (0.15 + 0.35 * want)))
                flow = taken / dur
                t1 = start - (min(0.05, max(0.0, 0.3 * gap)) if marked else min(0.05, 0.4 * gap))
                t0 = max((prev_end - 0.05) if marked else (prev_end + min(0.06, 0.3 * gap)),
                         t1 - dur, 0.01)
                if t1 - t0 >= 0.04 and taken > 0.02:
                    if self.breaths > 0:
                        amp = BREATH_AMP * self.breaths * float(np.clip((flow / 0.9) ** 0.7, 0.25, 2.2))
                        bright = float(np.clip((flow - 0.8) / 1.6, 0.0, 1.0))
                        self.breath_ev.append((t0, t1, amp, bright))
                else:
                    taken = 0.0
            if (taken >= 0.1 or marked) and stretch is not None:
                # a refill ends the stretch that ran dry
                if stretch[2] >= 0.75:
                    self.air_warnings.append(tuple(stretch))
                stretch = None
            if taken >= 0.3 or marked or k == 0:
                stretch_start = start             # the last real breath
            air = max(0.0, air) + taken
            # sing the phrase
            for u in ph:
                d = max(0.0, u.eff_off - u.c_start)
                rate = self._air_rate(u.c_start, u)
                if air - rate * d < 0.0:
                    short = (rate * d - max(0.0, air)) / rate
                    if stretch is None:
                        stretch = [stretch_start, u.c_start + max(0.0, air) / rate, 0.0]
                    stretch[2] += short
                air -= rate * d
            air = max(0.0, air)
            prev_end = ph[-1].eff_off
        if stretch is not None and stretch[2] >= 0.75:
            self.air_warnings.append(tuple(stretch))
        self.air_end = air

    # -------------------------------------------------------- rendering
    def render(self, gain_db: np.ndarray) -> np.ndarray:
        n = self.n
        N = int(self.dur * SR) + BLOCK
        # --- smooth control tracks (coarticulation: speed depends on the sound)
        W = self.SW
        for k in range(5):
            self.F[k] = _var_smooth(self.F[k], W)
            self.B[k] = _var_smooth(self.B[k], W)
        self.NZ = _var_smooth(self.NZ, W)
        # the tongue and jaw are never perfectly still: vowel colour wanders
        # by about a percent
        for k in range(3):
            self.F[k] *= 1.0 + 0.012 * self._lf_noise(n, 2.0, rate=CR)
        AV = _hann_smooth(self.AV, 8)
        AV *= 10 ** (0.4 * self._lf_noise(n, 1.5, rate=CR) / 20.0)   # ~0.4 dB swell/sag
        AH = _hann_smooth(self.AH, 8)
        for tc, w, depth in self.dips:
            i0, i1 = self._i(tc - w / 2), self._i(tc + w / 2)
            if i1 > i0:
                AV[i0:i1] *= 1.0 - depth * np.hanning(i1 - i0)

        midi = self._pitch_contour()
        if midi is None:
            return np.zeros(N, dtype=np.float32)
        idx = np.arange(n)
        # sopranos open F1 (and F2) above the pitch on high notes
        f0_ctrl = 440.0 * 2 ** ((midi - 69) / 12)
        self.F[0] = np.maximum(self.F[0], f0_ctrl * 1.08)
        self.F[1] = np.maximum(self.F[1], self.F[0] + 250)
        self.F[2] = np.maximum(self.F[2], self.F[1] + 300)
        opening = _hann_smooth(self.bloom, 8)       # how far the voice has opened
        AV = AV * self._level_comp(f0_ctrl, AV)

        # --- sample-rate tracks
        ts = np.arange(N) / SR
        fpos = ts * CR
        midi_s = np.interp(fpos, idx, midi)
        vib_s = np.interp(fpos, idx, _hann_smooth(self.vib, 60))
        cents_s = np.interp(fpos, idx, _hann_smooth(self.cents, 10))
        rate = np.interp(fpos, idx, _hann_smooth(self.vrate, 150))
        rate = rate * (1 + 0.04 * np.sin(2 * np.pi * 0.23 * ts + self.rng.uniform(0, 6)))
        vphase = np.cumsum(2 * np.pi * rate / SR) + self.rng.uniform(0, 2 * np.pi)
        drift = self._lf_noise(N, 1.2) * 5.0 + self._lf_noise(N, 4.0) * 1.5   # cents
        jitter = self._lf_noise(N, 60.0) * 3.0
        cents = vib_s * np.sin(vphase) + cents_s + drift + jitter
        f0 = 440.0 * 2 ** ((midi_s - 69 + cents / 100.0) / 12)
        phase = np.cumsum(f0 / SR) % 1.0

        AV_s = np.interp(fpos, idx, AV)
        AH_s = np.interp(fpos, idx, AH)
        open_s = np.interp(fpos, idx, opening)
        # higher in the voice is a little louder and brighter, lower a little
        # softer and darker
        dp = np.clip(midi - self.mid_pitch, -12.0, 12.0)
        lvl_c = gain_db + _hann_smooth(self.accent_db, 6) + np.clip(0.1 * dp, -1.5, 1.5)
        lvl = np.interp(fpos, idx, lvl_c)
        # voice quality follows loudness (soft = breathy, loud = pressed) and
        # blooms as the voice opens: darker at the very start of a note
        q = np.clip((lvl + 22) / 26, 0, 1)
        q = np.clip(q - 0.45 * (1.0 - open_s) + np.interp(fpos, idx, self.qoff + 0.008 * dp), 0, 1)
        glot = self._glottal(phase, f0, q)
        shimmer = 1 + 0.04 * self._lf_noise(N, 30.0)
        noise = self.rng.standard_normal(N)
        # breath noise is louder while the glottis is open
        open_mask = ((phase > _CLOSE_PHASE - 0.6) & (phase < _CLOSE_PHASE)).astype(float) * 0.7 + 0.3
        breath = noise * open_mask * (self.breathiness * (1.4 - 0.8 * q))
        x = AV_s * (glot * shimmer + breath) + AH_s * noise * 2.0

        y = self._cascade(x, AV_s, AH_s)
        y = sosfilt(_PRESENCE, y)
        y += self._frication(N)
        y += self._breath_noise(N)
        y *= 10 ** (lvl / 20.0)
        return y.astype(np.float32)

    def _level_comp(self, f0, AV, hop=10):
        """Per-frame gain that cancels the accident of a harmonic landing on
        (or between) formant peaks, so a vowel's loudness doesn't jump around
        with pitch.  Natural differences between vowels are kept."""
        amps = _glottal_tables()[("amps", _Q_LEVELS // 2)]
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

    def _lf_noise(self, N, hz, rate=SR):
        k = max(2, int(N / rate * hz) + 2)
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
        qpos = np.clip(q, 0.0, 1.0) * (_Q_LEVELS - 1)
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
            qp = qpos[s:e]
            lo_q = int(np.floor(qp.min()))
            hi_q = min(_Q_LEVELS - 1, int(np.ceil(qp.max())))
            fr = frac[s:e]
            acc = np.zeros(e - s)
            for qi in range(lo_q, hi_q + 1):
                wgt = np.clip(1.0 - np.abs(qp - qi), 0.0, 1.0)     # triangular cross-fade
                if not wgt.any():
                    continue
                t = tabs[qi][h]
                acc += wgt * (t[i0[s:e]] * (1 - fr) + t[i1[s:e]] * fr)
            out[s:e] = acc
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
            if b <= a or a >= N or b <= 0:
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
                env[:na] = np.sin(0.5 * np.pi * np.linspace(0, 1, na)) ** 2
            if nr < L:
                env[-nr:] *= np.linspace(1, 0, nr) ** 1.5
            # normalize the band-passed noise to unit RMS, then scale
            rms = np.sqrt(np.mean(nz ** 2)) + 1e-9
            skip = max(0, -a)          # an event that starts before t=0
            out[a + skip:b] += (nz / rms * env * amp * FRIC_GAIN)[skip:]
        return out

    def _breath_noise(self, N):
        out = np.zeros(N)
        for (t0, t1, amp, bright) in self.breath_ev:
            a, b = max(0, int(t0 * SR)), min(N, int(t1 * SR))
            L = b - a
            if L < 64:
                continue
            nz = self.rng.standard_normal(L + 512)
            lo = sosfilt(_BREATH_LO, nz)[512:]
            hi = sosfilt(_BREATH_HI, nz)[512:]
            # a gasp has more high hiss (narrower, faster airflow)
            sig = lo / (np.sqrt(np.mean(lo ** 2)) + 1e-9) + \
                (0.5 + 0.9 * bright) * hi / (np.sqrt(np.mean(hi ** 2)) + 1e-9)
            sig /= np.sqrt(np.mean(sig ** 2)) + 1e-9
            # airflow rises then tails off; a gasp peaks earlier and stops harder
            x = np.linspace(0, 1, L)
            pk = 0.6 - 0.25 * bright
            env = np.where(x < pk, np.sin(0.5 * np.pi * x / pk) ** 2,
                           np.clip(np.cos(0.5 * np.pi * (x - pk) / (1 - pk)), 0.0, 1.0) ** (2 - bright))
            env *= 1.0 + 0.15 * self._lf_noise(L, 12.0)        # turbulence flutter
            out[a:b] += sig * env * amp * FRIC_GAIN
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
_BREATH_LO = butter(2, [450, 2400], btype="band", fs=SR, output="sos")
_BREATH_HI = butter(2, [2400, 6500], btype="band", fs=SR, output="sos")

# Overall level of noise sources relative to the voiced cascade output.
# (Set by calibrate.py so that 's' sits ~10 dB under an 'ah'.)
FRIC_GAIN = 2.6
# An inhale before a phrase, roughly 25 dB under a sung 'ah'
BREATH_AMP = 0.05
