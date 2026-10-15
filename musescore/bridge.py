#!/usr/bin/env python3
"""Run by the MuseScore plugin: render the open score for synced playback."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acappella.musescore import main  # noqa: E402

if __name__ == "__main__":
    main()
