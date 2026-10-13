#!/usr/bin/env python3
"""Render an a cappella MuseScore/MusicXML score to MP3.

    python render.py "My Arrangement.mscz"
    python render.py song.mscz --measures 17-32 --focus Alto
    python render.py --help
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acappella.render import main  # noqa: E402

if __name__ == "__main__":
    main()
