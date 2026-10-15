#!/bin/bash
# Installs the MuseScore 4 integration (macOS):
#   - the "Acappella Render" VST3 effect, which plays the render in sync with
#     MuseScore's playback when it sits on the Master channel
#   - the "Acappella Render" MuseScore plugin, whose Render button renders the
#     open score for it
#
#   musescore/install.sh               build and install
#   musescore/install.sh --uninstall   remove both again
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
BUILD="$HERE/build"
SDK="$BUILD/vst3sdk"
DATA="$HOME/Library/Application Support/AcappellaRender"
VST="$HOME/Library/Audio/Plug-Ins/VST3/Acappella Render.vst3"
PLUGIN_DIR="$HOME/Documents/MuseScore4/Plugins/AcappellaRender"

if [ "${1:-}" = "--uninstall" ]; then
    rm -rf "$VST" "$PLUGIN_DIR"
    echo "Removed the effect and the plugin. Renders are still in $DATA"
    exit 0
fi

# a Python that can run the renderer
if [ -x "$REPO/.venv/bin/python" ]; then
    PY="$REPO/.venv/bin/python"
else
    PY="$(command -v python3)"
fi
if ! "$PY" -c "import numpy, scipy, cmudict" 2>/dev/null; then
    echo "$PY can't import numpy/scipy/cmudict; install the renderer's requirements first (see README)." >&2
    exit 1
fi

# the VST 3 SDK (MIT licensed), just the parts the effect needs
[ -d "$SDK/.git" ] || git clone --depth 1 https://github.com/steinbergmedia/vst3sdk.git "$SDK"
git -C "$SDK" submodule update --init --force --depth 1 base cmake pluginterfaces public.sdk

cmake -S "$HERE/vst" -B "$BUILD/vst" -DVST3_SDK_DIR="$SDK" >/dev/null
cmake --build "$BUILD/vst" -j 8

mkdir -p "$(dirname "$VST")" "$PLUGIN_DIR" "$DATA"
rm -rf "$VST"
cp -R "$BUILD/vst/VST3/Release/AcappellaRender.vst3" "$VST"
codesign --force --sign - "$VST" >/dev/null 2>&1 || true
# what MuseScore does for a new plugin it finds at startup
MSCORE="/Applications/MuseScore 4.app/Contents/MacOS/mscore"
if [ -x "$MSCORE" ]; then
    "$MSCORE" --register-audio-plugin "$VST" >/dev/null 2>&1 || true
fi

"$PY" - "$HERE/AcappellaRender.qml" "$PLUGIN_DIR/AcappellaRender.qml" "$PY" "$HERE/bridge.py" "$DATA" <<'EOF'
import json, sys
src, dst, py, bridge, data = sys.argv[1:]
qml = open(src, encoding="utf-8").read()
for key, val in (("PYTHON", py), ("BRIDGE", bridge), ("DATA", data)):
    qml = qml.replace(f'"@{key}@"', json.dumps(val, ensure_ascii=False))
open(dst, "w", encoding="utf-8").write(qml)
EOF

cat <<MSG

Installed:
  effect: $VST
  plugin: $PLUGIN_DIR

Next, in MuseScore 4:
  1. Quit and reopen MuseScore so it finds the new effect.
  2. Home > Plugins: enable "Acappella Render" (it then appears in the Plugins menu).
  3. Open a score, choose Plugins > Acappella Render, and press Render.
  4. Once per score: open the Mixer and add "Acappella Render" to the Master
     channel's Audio FX. Press Play: you hear the render, following the cursor.
     Switch the effect off to hear MuseScore's own sounds again.
MSG
