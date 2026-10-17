"""Rendered voices kept between runs, so a re-render only sings what changed.

Each singer's track is stored under a hash of everything that goes into it:
the syllables and notes it sings, its dynamics curve, the singer's settings,
the length of the render, and the synthesizer's own code (so changing the
synthesizer never reuses a stale voice).  Mixing (balance, panning, reverb)
happens afterwards, so a voice is reused even when only those change.
Tracks are stored exactly as a fresh render produces them (a voice is
float32; a part's percussion track, kept the same way, float64).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import threading
from pathlib import Path

import numpy as np

CACHE_VERSION = 2
LIMIT_BYTES = 3 * 1024 ** 3        # oldest voices are dropped beyond this


def cache_dir() -> Path:
    env = os.environ.get("ACAPPELLA_CACHE")
    if env:
        return Path(env)
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "acappella-render"


_CODE = None


def _code_fingerprint() -> bytes:
    global _CODE
    if _CODE is None:
        import scipy

        here = Path(__file__).resolve().parent
        from . import synth

        fast = "numba" if synth._fast is not None else "numpy"
        h = hashlib.sha256(f"v{CACHE_VERSION} np{np.__version__} sp{scipy.__version__} {fast}".encode())
        for name in ("synth.py", "fastsynth.py", "perc.py", "render.py"):
            h.update((here / name).read_bytes())
        _CODE = h.digest()
    return _CODE


def code_fingerprint() -> bytes:
    """Changes with the synthesizer's and the mixer's code (and numpy's and scipy's)."""
    return _code_fingerprint()


def track_path(key: str) -> Path:
    """Where the track cached under key is (a .npy file)."""
    return cache_dir() / f"{key}.npy"


def voice_key(job: dict) -> str:
    """Hash of a voice's (or a percussion track's) inputs.  pickle writes the units, arrays and numbers
    exactly and in order (the planner builds them the same way every run)."""
    h = hashlib.sha256(_code_fingerprint())
    h.update(pickle.dumps(job, protocol=5))
    return h.hexdigest()


def part_key(voice_keys, gain) -> str:
    """Hash for a value computed from a part's voices and dynamics curve."""
    h = hashlib.sha256(b"part" + "".join(voice_keys).encode())
    h.update(np.ascontiguousarray(gain).tobytes())
    return h.hexdigest()


_VALUES = None
MAX_VALUES = 4000


def _values_file() -> Path:
    return cache_dir() / "values.json"


def load_value(key: str):
    global _VALUES
    if _VALUES is None:
        try:
            _VALUES = json.loads(_values_file().read_text())
        except (OSError, ValueError):
            _VALUES = {}
    return _VALUES.get(key)


def store_value(key: str, value) -> None:
    if load_value(key) == value:
        return
    _VALUES[key] = value
    while len(_VALUES) > MAX_VALUES:
        del _VALUES[next(iter(_VALUES))]           # oldest first
    try:
        cache_dir().mkdir(parents=True, exist_ok=True)
        tmp = cache_dir() / f".values.{os.getpid()}.json"
        tmp.write_text(json.dumps(_VALUES))
        os.replace(tmp, _values_file())
    except OSError:
        pass


def load(key: str):
    """(track, air warnings) of a cached voice, or None.  The file is read
    in one go: a voice not used for a while is often no longer in memory,
    and reading it through a memory map, page by page, is several times
    slower then."""
    d = cache_dir()
    npy, meta = d / f"{key}.npy", d / f"{key}.json"
    try:
        ys = np.load(npy)
        warnings = [tuple(w) for w in json.loads(meta.read_text())]
        os.utime(npy)                       # recently used: keep it longest
    except (OSError, ValueError):
        return None
    return ys, warnings


def store(key: str, ys: np.ndarray, warnings) -> bool:
    """Keep a voice; False if it couldn't be written (the cache is then
    just slower)."""
    d = cache_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{key}.{os.getpid()}.{threading.get_ident()}.npy"
        np.save(tmp, ys)
        (d / f"{key}.json").write_text(json.dumps([list(w) for w in warnings]))
        os.replace(tmp, d / f"{key}.npy")
        return True
    except OSError:
        return False


def prune(limit: int = LIMIT_BYTES) -> None:
    d = cache_dir()
    try:
        files = sorted(d.glob("*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return
    total = 0
    for p in files:
        try:
            total += p.stat().st_size
            if total > limit:
                p.unlink(missing_ok=True)
                p.with_suffix(".json").unlink(missing_ok=True)
        except OSError:
            pass
