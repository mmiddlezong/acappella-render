"""Vocal percussion: the snare, kick and hi-hat hits a singer makes on
x-noteheads.  Each hit is built from the parts of the real sound (the crack
of the attack, the drum body, the buzz of the wires) with a little mouth
colour, so it reads as a percussion hit rather than a sung consonant."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from .synth import CR, SR

# loudness of a hit's loudest 50 ms against a sung part at the same dynamic
KIND_LEVEL = {"snare": 2.0, "kick": 2.4, "hat": 0.8}
REF = 0.12                       # a sung part's calibrated loudness (render.py)


def _band(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype="band", fs=SR, output="sos"), x)


def _hp(x, fc, order=2):
    return sosfilt(butter(order, fc, btype="high", fs=SR, output="sos"), x)


def _rise(t, ms):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t / (ms / 1000.0), 0.0, 1.0))


def _fade(x, frac=0.35):
    """Fade the tail to silence so a hit never ends on a cut."""
    n = int(len(x) * frac)
    x = x.copy()
    x[-n:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(n) / n)
    return x


def _norm(x):
    """Scale so the loudest 50 ms has RMS 1."""
    n = int(0.05 * SR)
    p = np.convolve(x ** 2, np.ones(n) / n, mode="valid")
    return x / np.sqrt(max(float(p.max()), 1e-12))


def snare(rng, tone=1.0) -> np.ndarray:
    t = np.arange(int(0.42 * SR)) / SR
    # attack crack: a very short, bright noise burst (the stick, or the
    # tongue/lip release of a mouth snare)
    crack = _band(rng.standard_normal(len(t)), 1500, 7000) * np.exp(-t / 0.006) * _rise(t, 0.4)
    # drum body: two head modes whose pitch drops as the hit settles
    f0 = 185.0 * tone
    f = f0 * (1.0 + 0.4 * np.exp(-t / 0.007))
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = (np.sin(ph) + 0.5 * np.sin(1.59 * ph + 0.9)) * np.exp(-t / 0.045) * _rise(t, 1.0)
    # wires: a broadband buzz that rings past the body, with a mouth
    # resonance around 2-3 kHz
    nz = rng.standard_normal(len(t))
    wires = _band(nz, 900, 10000) + 0.8 * _band(nz, 1900, 3200)
    wires *= (0.72 * np.exp(-t / 0.05) + 0.28 * np.exp(-t / 0.13)) * _rise(t, 1.5)
    y = 1.0 * _norm(wires) + 0.55 * _norm(body) + 0.45 * _norm(crack)
    return _fade(_norm(y))


def kick(rng, tone=1.0) -> np.ndarray:
    t = np.arange(int(0.4 * SR)) / SR
    f = 50.0 * tone + 75.0 * np.exp(-t / 0.025)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.tanh(1.6 * np.sin(ph)) * np.exp(-t / 0.09) * _rise(t, 2.0)
    pop = _band(rng.standard_normal(len(t)), 600, 3500) * np.exp(-t / 0.004) * _rise(t, 0.3)
    return _fade(_norm(_norm(body) + 0.3 * _norm(pop)))


def hat(rng, tone=1.0, open_=False) -> np.ndarray:
    t = np.arange(int((0.3 if open_ else 0.15) * SR)) / SR
    nz = rng.standard_normal(len(t))
    tick = _hp(nz, 4000) * np.exp(-t / 0.003)
    hiss = _band(nz, 5500 * tone, 12000, order=3)
    hiss *= np.exp(-t / (0.09 if open_ else 0.024)) * _rise(t, 0.8)
    return _fade(_norm(0.5 * _norm(tick) + _norm(hiss)))


def render_hits(units, gain_db: np.ndarray, n: int, seed: int = 0, humanize: float = 1.0):
    """One mono track with every percussion hit in `units`, at the part's
    dynamic (plus accents), on a scale where a sung part's loudness at 'f'
    is REF."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for u in units:
        kind = u.perc
        tone = float(np.exp(rng.normal(0.0, 0.03)))
        if kind == "kick":
            y = kick(rng, tone)
        elif kind == "hat":
            y = hat(rng, tone)
        else:
            kind = "snare"
            y = snare(rng, tone)
        t = u.t_on + rng.normal(0.0, 0.004) * humanize
        i_ctl = int(np.clip(round(t * CR), 0, len(gain_db) - 1))
        db = float(gain_db[i_ctl]) + max(0.0, u.accent) + rng.normal(0.0, 0.7) * humanize
        lvl = REF * KIND_LEVEL[kind] * 10 ** (db / 20.0)
        # round off the very first peak (as a mic preamp or compressor would),
        # so one hit doesn't set the level of the whole mix
        y = 2.5 * lvl * np.tanh(y / 2.5)
        i = max(0, int(round(t * SR)))
        m = min(len(y), n - i)
        if m > 0:
            out[i:i + m] += y[:m]
    return out
