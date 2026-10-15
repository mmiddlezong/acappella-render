"""Rendered voices kept between runs, so a re-render only sings what changed.

Each singer's track is stored under a hash of everything that goes into it:
the syllables and notes it sings, its dynamics curve, the singer's settings,
the length of the render, and the synthesizer's own code (so changing the
synthesizer never reuses a stale voice).  Mixing (balance, panning, reverb)
happens afterwards, so a voice is reused even when only those change.
Tracks are float32, exactly what a fresh render would produce.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

CACHE_VERSION = 1
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
        h = hashlib.sha256(f"v{CACHE_VERSION} np{np.__version__} sp{scipy.__version__}".encode())
        for name in ("synth.py", "render.py"):
            h.update((here / name).read_bytes())
        _CODE = h.digest()
    return _CODE


def _feed(h, obj):
    """Hash a value by content (dataclasses, lists, arrays, numbers, text)."""
    if isinstance(obj, np.ndarray):
        h.update(f"A{obj.dtype}{obj.shape}".encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        h.update(f"D{type(obj).__name__}".encode())
        for f in dataclasses.fields(obj):
            h.update(f.name.encode())
            _feed(h, getattr(obj, f.name))
    elif isinstance(obj, (list, tuple)):
        h.update(f"L{type(obj).__name__}{len(obj)}".encode())
        for x in obj:
            _feed(h, x)
    elif isinstance(obj, dict):
        h.update(f"M{len(obj)}".encode())
        for k in sorted(obj, key=repr):
            _feed(h, k)
            _feed(h, obj[k])
    elif isinstance(obj, float):
        h.update(b"F" + obj.hex().encode())
    elif isinstance(obj, (bool, int, str, type(None))):
        h.update(f"{type(obj).__name__}:{obj!r}".encode())
    else:
        raise TypeError(f"can't fingerprint {type(obj).__name__}")


def voice_key(job: dict) -> str:
    h = hashlib.sha256(_code_fingerprint())
    _feed(h, job)
    return h.hexdigest()


def load(key: str):
    """(track, air warnings) of a cached voice, or None."""
    d = cache_dir()
    npy, meta = d / f"{key}.npy", d / f"{key}.json"
    try:
        ys = np.load(npy)
        warnings = [tuple(w) for w in json.loads(meta.read_text())]
        os.utime(npy)                       # recently used: keep it longest
    except (OSError, ValueError):
        return None
    return ys, warnings


def store(key: str, ys: np.ndarray, warnings) -> None:
    d = cache_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{key}.{os.getpid()}.npy"
        np.save(tmp, np.asarray(ys, dtype=np.float32))
        (d / f"{key}.json").write_text(json.dumps([list(w) for w in warnings]))
        os.replace(tmp, d / f"{key}.npy")
    except OSError:
        pass                                # a cache that can't be written is just slower


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
