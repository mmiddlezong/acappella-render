"""A render server that stays running between renders from MuseScore.

Starting a render from scratch costs about a second before any singing
starts: Python, numpy, scipy and numba load, the compiled kernels and the
wavetables are set up, and worker processes start.  The bridge (see
musescore.py) therefore hands its renders to this server, which keeps all
of that loaded, along with a pool of warmed-up worker processes that sing
voices in parallel.  The first render starts it; after IDLE_EXIT seconds
without a render it exits.  When the renderer's code changes it restarts.

Requests arrive over a Unix socket only the user can open: one JSON line
{"argv": [...bridge arguments...], "code": "<fingerprint>"}; the reply is
{"ok": true}, {"error": "..."} or {"restart": true}.

This module imports nothing heavy at the top: the bridge imports it to
send requests and needs to start quickly.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

IDLE_EXIT = 20 * 60
HERE = Path(__file__).resolve().parent


def data_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "AcappellaRender"


def socket_path() -> str:
    p = str(data_dir() / "render.sock")
    if len(p.encode()) >= 100:            # too long for a Unix socket address
        p = f"/tmp/acappella-render-{os.getuid()}.sock"
    return p


def code_fingerprint() -> str:
    """Changes whenever any of the renderer's source files do."""
    parts = []
    for f in sorted(HERE.glob("*.py")):
        st = f.stat()
        parts.append(f"{f.name}:{st.st_mtime_ns}:{st.st_size}")
    return "|".join(parts)


# ------------------------------------------------------------------ client
def _send(argv, timeout_connect=1.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout_connect)
    s.connect(socket_path())
    s.settimeout(None)                     # a render takes as long as it takes
    with s:
        s.sendall((json.dumps({"argv": argv, "code": code_fingerprint()}) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                raise ConnectionError("render server closed the connection")
            data += chunk
    return json.loads(data)


def _can_connect():
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(socket_path())
        s.close()
        return True
    except OSError:
        return False


def _start_server():
    import fcntl

    # one starter at a time, so two renders at once don't start two servers
    with open(data_dir() / "server.start.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _can_connect():
            return True
        log = open(data_dir() / "server.log", "a")
        subprocess.Popen([sys.executable, "-m", "acappella.server"], cwd=str(HERE.parent),
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        deadline = time.time() + 60
        while time.time() < deadline:
            time.sleep(0.05)
            if _can_connect():
                return True
    return False


def render_via_server(argv) -> bool:
    """Have the server run the bridge with argv (starting it, or a fresh one
    after a code change, if needed).  False if it can't be used: the caller
    renders by itself then."""
    data_dir().mkdir(parents=True, exist_ok=True)
    for _attempt in range(2):
        try:
            reply = _send(argv)
        except (OSError, ConnectionError):
            if not _start_server():
                return False
            continue
        if reply.get("restart"):
            time.sleep(0.2)                # it exits; the next attempt starts a new one
            if not _start_server():
                return False
            continue
        if "error" in reply:
            raise SystemExit(reply["error"])
        return True
    return False


# ------------------------------------------------------------------ server
def serve():
    from . import musescore, render

    path = socket_path()
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    code = code_fingerprint()
    old_umask = os.umask(0o077)             # only this user can connect
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    os.umask(old_umask)
    srv.listen(8)
    render.keep_pool(wait=False)    # worker processes kept between renders, warming up meanwhile
    print(time.strftime("%H:%M:%S"), "render server ready", flush=True)

    last = [time.time()]
    busy = [0]
    lock = threading.Lock()

    def handle(conn):
        with conn:
            f = conn.makefile("rb")
            try:
                req = json.loads(f.readline())
            except ValueError:
                return
            if req.get("code") != code:
                conn.sendall(b'{"restart": true}\n')
                print(time.strftime("%H:%M:%S"), "code changed: restarting", flush=True)
                render.close_pool()
                os._exit(0)
            with lock:
                busy[0] += 1
            try:
                musescore.main(req.get("argv", []))
                reply = {"ok": True}
            except SystemExit as e:
                reply = {"error": str(e)} if e.code not in (None, 0) else {"ok": True}
            except BaseException as e:  # noqa: BLE001 - the bridge already wrote the status
                reply = {"error": f"{type(e).__name__}: {e}"}
            finally:
                with lock:
                    busy[0] -= 1
                    last[0] = time.time()
            try:
                conn.sendall((json.dumps(reply) + "\n").encode())
            except OSError:
                pass

    srv.settimeout(30)
    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            with lock:
                idle = busy[0] == 0 and time.time() - last[0] > IDLE_EXIT
            if idle:
                print(time.strftime("%H:%M:%S"), "idle: exiting", flush=True)
                break
            continue
        threading.Thread(target=handle, args=(conn,), daemon=True).start()
    srv.close()
    try:
        os.unlink(path)
    except OSError:
        pass
    render.close_pool()
    os._exit(0)


if __name__ == "__main__":
    serve()
