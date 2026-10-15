"""Bridge between the MuseScore plugin and the renderer.

The plugin (musescore/AcappellaRender.qml) saves the score and runs this.
It renders the saved .mscz (on MuseScore's own tempo map, so it lines up
with MuseScore's playback) and points the Acappella Render effect on
MuseScore's master bus at the result, via a small manifest file:

    ~/Library/Application Support/AcappellaRender/current.txt

Progress goes to status.json next to it, which the plugin's dialog polls.
The settings of the last render are kept in settings.json, so the one-click
Render button can render the same way again.  Voices that didn't change
since an earlier render are reused (see cache.py), so a re-render after an
edit only sings the parts that changed.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.parse
from pathlib import Path

DATA = Path.home() / "Library" / "Application Support" / "AcappellaRender"
MANIFEST = DATA / "current.txt"
STATUS = DATA / "status.json"
SETTINGS = DATA / "settings.json"
DEFAULTS = {"singers": 2, "neutral": "", "musescore_gain": 0.0}
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


def load_settings() -> dict:
    out = dict(DEFAULTS)
    try:
        saved = json.loads(SETTINGS.read_text(encoding="utf-8"))
        out.update({k: type(DEFAULTS[k])(saved[k]) for k in DEFAULTS if k in saved})
    except (OSError, ValueError, TypeError):
        pass
    return out


def save_settings(settings: dict):
    _write_atomic(SETTINGS, json.dumps({k: settings[k] for k in DEFAULTS}))


def notify(message: str):
    """A macOS notification (for renders started without the dialog open)."""
    if sys.platform != "darwin":
        return
    subprocess.run(["osascript", "-e", "on run argv", "-e",
                    'display notification (item 1 of argv) with title "Acappella Render"',
                    "-e", "end run", message], check=False, capture_output=True)


def set_mix(musescore_gain: float):
    settings = load_settings()
    settings["musescore_gain"] = musescore_gain
    save_settings(settings)
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
    # settings left out are the ones the last render used
    p.add_argument("--singers", type=int)
    p.add_argument("--neutral", help="sing lyric words on this syllable ('' = off)")
    p.add_argument("--musescore-gain", type=float,
                   help="how much of MuseScore's own sound to keep under the render")
    p.add_argument("--mix-only", action="store_true",
                   help="just change --musescore-gain for the current render")
    p.add_argument("--notify", action="store_true",
                   help="post a notification when the render is done or fails")
    p.add_argument("--url", help="an acappella-render://render?score=NAME link (from the "
                                 "toolbar button, via the helper app)")
    # options from an older plugin (still loaded in a running MuseScore) are ignored
    a, _unknown = p.parse_known_args(argv)
    if a.url:
        u = urllib.parse.urlsplit(a.url)
        names = urllib.parse.parse_qs(u.query).get("score", [])
        if u.scheme != "acappella-render" or u.netloc != "render" or len(names) != 1:
            raise SystemExit(f"not a render link: {a.url}")
        a.score_name = names[0]
        a.request = a.request or f"toolbar-{int(time.time() * 1000)}"
    DATA.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    for k in DEFAULTS:
        if getattr(a, k) is not None:
            settings[k] = getattr(a, k)
    if a.mix_only:
        set_mix(settings["musescore_gain"])
        return
    save_settings(settings)
    with open(DATA / "bridge.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:                           # one render at a time
            _status(state="rendering", request=a.request, score=a.score_name, started=time.time(),
                    message="Waiting for the render before it to finish...")
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            render_for_musescore(a.score_name, a.request, singers=settings["singers"],
                                 neutral=settings["neutral"],
                                 musescore_gain=settings["musescore_gain"])
        except BaseException as e:  # noqa: BLE001 - report everything to the dialog
            quit_ = isinstance(e, SystemExit)
            msg = str(e) if quit_ else f"{type(e).__name__}: {e}"
            _status(state="error", request=a.request, score=a.score_name, finished=time.time(),
                    message=msg or "Render failed",
                    notes=[] if quit_ else traceback.format_exc().splitlines()[-8:])
            if a.notify:
                notify(msg or "Render failed")
            raise
        if a.notify:
            st = json.loads(STATUS.read_text(encoding="utf-8"))
            notify(st.get("message", "Rendered") + ". Press Play.")
