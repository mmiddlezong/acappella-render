#!/usr/bin/env python3
"""Run by the MuseScore plugin: render the open score for synced playback.

The render itself happens in the render server (acappella/server.py), which
stays loaded between renders; this starts it when needed and falls back to
rendering here if it can't be reached.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    argv = sys.argv[1:]
    done = False
    if "--mix-only" not in argv and not os.environ.get("ACAPPELLA_NO_SERVER"):
        from acappella.server import render_via_server  # noqa: E402
        done = render_via_server(argv)
    if not done:
        from acappella.musescore import main  # noqa: E402
        main(argv)
