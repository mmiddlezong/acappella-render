"""Vocal percussion: the snare, kick and hi-hat hits a singer makes with
their mouth on x-noteheads.  Each hit is a consonant-like release (lips or
tongue popping open) followed by a short burst of breath shaped by the
mouth, the way a beatboxer's "pf", "b" or "ts" is made.  The sound stops
when the singer has to start the next sung note."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfilt

from .synth import CR, SR

# loudness of a hit's loudest 50 ms against a sung part at the same dynamic
KIND_LEVEL = {"snare": 1.0, "kick": 1.2, "hat": 0.45}
REF = 0.12                       # a sung part's calibrated loudness (render.py)


@lru_cache(maxsize=256)
def _band_sos(order, lo, hi):
    return butter(order, [lo, hi], btype="band", fs=SR, output="sos")


def _band(x, lo, hi, order=2):
    return sosfilt(_band_sos(order, lo, hi), x)


def _res(x, fc, bw):
    """A mouth resonance: a single 2-pole band-pass around fc."""
    lo, hi = max(60.0, fc - bw / 2), min(0.45 * SR, fc + bw / 2)
    return sosfilt(butter(1, [lo, hi], btype="band", fs=SR, output="sos"), x)


def _rise(t, ms):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t / (ms / 1000.0), 0.0, 1.0))


def _flutter(rng, n, hz, depth):
    """Turbulence: the breath noise of a real mouth wobbles in level."""
    k = max(2, int(n / SR * hz) + 2)
    pts = rng.standard_normal(k)
    return 1.0 + depth * np.interp(np.linspace(0, k - 1, n), np.arange(k), pts)


def _fade(x, frac=0.35):
    """Fade the tail to silence so a hit never ends on a cut."""
    n = int(len(x) * frac)
    x = x.copy()
    x[-n:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(n) / n)
    return x


def _norm(x):
    """Scale so the loudest 50 ms has RMS 1."""
    n = min(len(x), int(0.05 * SR))
    c = np.cumsum(x ** 2)
    p = (c[n - 1:] - np.concatenate(([0.0], c[:-n]))) / n     # every 50 ms window's mean
    return x / np.sqrt(max(float(p.max()), 1e-12))


def snare(rng, tone=1.0) -> np.ndarray:
    """A mouth snare ("pf"/"psh"): a sharp pop of the lips, then a short
    puff of breath through the half-closed mouth, with a little thump from
    the cheeks."""
    t = np.arange(int(0.22 * SR)) / SR
    nz = rng.standard_normal(len(t))
    pop = _band(rng.standard_normal(len(t)), 500, 5000) * np.exp(-t / 0.0018) * _rise(t, 0.15)
    j = lambda: tone * float(np.exp(rng.normal(0.0, 0.04)))
    puff = (0.6 * _res(nz, 1500 * j(), 700) + 1.0 * _res(nz, 2800 * j(), 1000)
            + 0.75 * _res(nz, 4800 * j(), 2200) + 0.3 * _band(nz, 1000, 9000))
    puff *= (0.75 * np.exp(-t / 0.026) + 0.25 * np.exp(-t / 0.065)) * _rise(t, 1.5)
    puff *= _flutter(rng, len(t), 70.0, 0.3)
    f = 150.0 * tone * (1.0 + 0.3 * np.exp(-t / 0.006))
    thump = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.012) * _rise(t, 1.0)
    y = 1.0 * _norm(puff) + 0.6 * _norm(pop) + 0.3 * _norm(thump)
    return _fade(_norm(y), 0.3)


def kick(rng, tone=1.0) -> np.ndarray:
    """A mouth kick ("b"): the lips burst open over a short, low thump."""
    t = np.arange(int(0.2 * SR)) / SR
    f = 55.0 * tone + 45.0 * np.exp(-t / 0.02)
    body = np.tanh(1.5 * np.sin(2 * np.pi * np.cumsum(f) / SR)) * np.exp(-t / 0.05) * _rise(t, 1.5)
    pop = _band(rng.standard_normal(len(t)), 150, 2200) * np.exp(-t / 0.003) * _rise(t, 0.2)
    return _fade(_norm(_norm(body) + 0.45 * _norm(pop)), 0.35)


def hat(rng, tone=1.0) -> np.ndarray:
    """A mouth hi-hat ("ts"): a tongue tick, then a short hiss."""
    t = np.arange(int(0.11 * SR)) / SR
    nz = rng.standard_normal(len(t))
    tick = _band(nz, 3000, 9000) * np.exp(-t / 0.0015)
    hiss = _band(nz, 5200 * tone, 11000, order=3) * np.exp(-t / 0.02) * _rise(t, 1.0)
    hiss *= _flutter(rng, len(t), 90.0, 0.25)
    return _fade(_norm(0.5 * _norm(tick) + _norm(hiss)), 0.35)


def render_hits(units, gain_db: np.ndarray, n: int, seed: int = 0, humanize: float = 1.0,
                until=None):
    """One mono track with every percussion hit in `units`, at the part's
    dynamic (plus accents), on a scale where a sung part's loudness at 'f'
    is REF.  until[k] is when hit k must be silent again (the singer's next
    sound starts): a mouth can't hold a snare's tail and sing at once."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for k, u in enumerate(units):
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
        if until is not None and until[k] is not None:
            room = int(max(0.035, until[k] - t) * SR)
            if room < len(y):
                f = min(room // 2, int(0.012 * SR))
                y = y[:room].copy()
                y[-f:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(f) / f)
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
