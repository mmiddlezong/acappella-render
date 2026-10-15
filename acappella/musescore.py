"""Bridge between the MuseScore plugin and the renderer.

The plugin (musescore/AcappellaRender.qml) saves the score and runs this.
It renders the saved .mscz (on MuseScore's own tempo map, so it lines up
with MuseScore's playback) and points the Acappella Render effect on
MuseScore's master bus at the result, via a small manifest file:

    ~/Library/Application Support/AcappellaRender/current.txt

Progress goes to status.json next to it, which the plugin's dialog polls.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
import json
import os
import time
import traceback
from pathlib import Path

DATA = Path.home() / "Library" / "Application Support" / "AcappellaRender"
MANIFEST = DATA / "current.txt"
STATUS = DATA / "status.json"
RECENT = Path.home() / "Library" / "Application Support" / "MuseScore" / "MuseScore4" / "recent_files.json"
KEEP_RENDERS = 3


def _write_atomic(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _status(**kw):
    _write_atomic(STATUS, json.dumps(kw, ensure_ascii=False))


def _read_manifest() -> dict:
    out = {}
    try:
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition(" ")
            if k:
                out[k] = v
    except OSError:
        pass
    return out


def _write_manifest(m: dict):
    order = ["audio", "offset", "render_gain", "musescore_gain", "title", "score"]
    keys = order + [k for k in m if k not in order]
    _write_atomic(MANIFEST, "".join(f"{k} {m[k]}\n" for k in keys if k in m))


def find_score(name: str) -> Path:
    """The saved score MuseScore has open under this name: MuseScore's
    recent-files list has its path; the newest file by that name wins."""
    found = []
    try:
        entries = json.loads(RECENT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        entries = []
    for e in entries:
        p, shown = (e, None) if isinstance(e, str) else (e.get("path"), e.get("displayName"))
        if not p:
            continue
        p = Path(p)
        if p.suffix.lower() in (".mscz", ".mscx") and name in (p.stem, shown) and p.exists():
            found.append(p)
    if not found:
        raise SystemExit(f"Couldn't find the file for '{name}'. Save it as a .mscz first.")
    return max(found, key=lambda p: p.stat().st_mtime)


def set_mix(musescore_gain: float):
    m = _read_manifest()
    if "audio" in m:
        m["musescore_gain"] = f"{musescore_gain:g}"
        _write_manifest(m)


def render_for_musescore(name: str, request: str, singers=2, neutral=None, musescore_gain=0.0):
    from .render import PREROLL, render_score

    started = time.time()
    _status(state="rendering", request=request, score=name, started=started,
            message=f"Rendering {name}...")
    path = find_score(name)
    out_dir = DATA / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in path.stem)[:60]
    out = out_dir / f"{safe}-{int(started)}.wav"
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        render_score(path, out=out, singers=singers, neutral=neutral or None)
    _write_manifest({"audio": str(out), "offset": f"{PREROLL:g}", "render_gain": "1",
                     "musescore_gain": f"{musescore_gain:g}", "title": name, "score": str(path)})
    for old in sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)[:-KEEP_RENDERS]:
        old.unlink(missing_ok=True)
    lines = log.getvalue().splitlines()
    _status(state="done", request=request, score=name, path=str(path), started=started,
            finished=time.time(), message=(lines[0] if lines else "Rendered").split(" -> ")[0],
            notes=[ln.strip() for ln in lines[1:]])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--score-name", help="the score's name in MuseScore (its file name)")
    p.add_argument("--request", default="", help="id echoed back in status.json")
    p.add_argument("--singers", type=int, default=2)
    p.add_argument("--neutral", default="", help="sing lyric words on this syllable")
    p.add_argument("--musescore-gain", type=float, default=0.0,
                   help="how much of MuseScore's own sound to keep under the render")
    p.add_argument("--mix-only", action="store_true",
                   help="just change --musescore-gain for the current render")
    a = p.parse_args(argv)
    DATA.mkdir(parents=True, exist_ok=True)
    if a.mix_only:
        set_mix(a.musescore_gain)
        return
    with open(DATA / "bridge.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)          # one render at a time
        try:
            render_for_musescore(a.score_name, a.request, singers=a.singers, neutral=a.neutral,
                                 musescore_gain=a.musescore_gain)
        except BaseException as e:  # noqa: BLE001 - report everything to the dialog
            quit_ = isinstance(e, SystemExit)
            msg = str(e) if quit_ else f"{type(e).__name__}: {e}"
            _status(state="error", request=a.request, score=a.score_name, finished=time.time(),
                    message=msg or "Render failed",
                    notes=[] if quit_ else traceback.format_exc().splitlines()[-8:])
            raise
