#!/bin/bash
# Installs the MuseScore 4 integration (macOS):
#   - the "Acappella Render" VST3 effect, which plays the render in sync with
#     MuseScore's playback when it sits on the Master channel
#   - the "Acappella Render" MuseScore extension: a one-click Render button and
#     a Settings and status window on the Extensions toolbar of the Score page
#   - a small background helper app that the Render button's
#     acappella-render:// links start renders through
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
MS_DATA="$HOME/Library/Application Support/MuseScore/MuseScore4"
EXT_DIR="$MS_DATA/extensions/acappella_render"
OLD_PLUGIN="$HOME/Documents/MuseScore4/Plugins/AcappellaRender"    # where 1.0 installed
HELPER="$DATA/Acappella Render Helper.app"
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister

if [ "${1:-}" = "--uninstall" ]; then
    [ -d "$HELPER" ] && "$LSREGISTER" -u "$HELPER" 2>/dev/null || true
    rm -rf "$VST" "$EXT_DIR" "$OLD_PLUGIN" "$HELPER"
    echo "Removed the effect and the extension. Renders are still in $DATA"
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

mkdir -p "$(dirname "$VST")" "$EXT_DIR" "$DATA"
rm -rf "$VST"
cp -R "$BUILD/vst/VST3/Release/AcappellaRender.vst3" "$VST"
codesign --force --sign - "$VST" >/dev/null 2>&1 || true
# what MuseScore does for a new plugin it finds at startup
MSCORE="/Applications/MuseScore 4.app/Contents/MacOS/mscore"
if [ -x "$MSCORE" ]; then
    "$MSCORE" --register-audio-plugin "$VST" >/dev/null 2>&1 || true
fi

rm -rf "$OLD_PLUGIN"
"$PY" - "$HERE/extension" "$EXT_DIR" "$PY" "$HERE/bridge.py" "$DATA" "$MS_DATA/extensions/config.json" <<'EOF'
import json, shutil, sys
from pathlib import Path
src, dst, py, bridge, data, config = sys.argv[1:]
for f in ("manifest.json", "AcappellaRender.qml", "render.js"):
    text = (Path(src) / f).read_text(encoding="utf-8")
    for key, val in (("PYTHON", py), ("BRIDGE", bridge), ("DATA", data)):
        text = text.replace(f'"@{key}@"', json.dumps(val, ensure_ascii=False))
    (Path(dst) / f).write_text(text, encoding="utf-8")
# enabled, as if switched on in Home > Plugins (other extensions are kept as they are)
uri = json.loads((Path(src) / "manifest.json").read_text())["uri"]
try:
    entries = json.loads(Path(config).read_text(encoding="utf-8"))
except (OSError, ValueError):
    entries = []
entries = [e for e in entries if e.get("uri") not in
           (uri, "musescore://extensions/v1/acappellarender/acappellarender.qml")]
entries.append({"uri": uri, "actions": [{"code": "render", "exec_point": "manually"},
                                        {"code": "settings", "exec_point": "manually"}]})
Path(config).write_text(json.dumps(entries, indent=2), encoding="utf-8")
EOF

# the helper app: opens acappella-render://render?score=NAME and starts the render
"$PY" - "$HERE/helper.applescript" "$BUILD/helper.applescript" "$PY" "$HERE/bridge.py" "$DATA/bridge.log" <<'EOF'
import sys
src, dst, py, bridge, log = sys.argv[1:]
text = open(src, encoding="utf-8").read()
for key, val in (("PYTHON", py), ("BRIDGE", bridge), ("LOG", log)):
    text = text.replace(f'"@{key}@"', '"' + val.replace("\\", "\\\\").replace('"', '\\"') + '"')
open(dst, "w", encoding="utf-8").write(text)
EOF
[ -d "$HELPER" ] && "$LSREGISTER" -u "$HELPER" 2>/dev/null || true
rm -rf "$HELPER"
osacompile -o "$HELPER" "$BUILD/helper.applescript"
PLIST="$HELPER/Contents/Info.plist"
PB=/usr/libexec/PlistBuddy
$PB -c "Delete :CFBundleIdentifier" "$PLIST" 2>/dev/null || true
$PB -c "Add :CFBundleIdentifier string com.github.acappella-render.helper" \
    -c "Add :LSUIElement bool true" \
    -c "Add :CFBundleURLTypes array" \
    -c "Add :CFBundleURLTypes:0 dict" \
    -c "Add :CFBundleURLTypes:0:CFBundleURLName string AcappellaRender" \
    -c "Add :CFBundleURLTypes:0:CFBundleURLSchemes array" \
    -c "Add :CFBundleURLTypes:0:CFBundleURLSchemes:0 string acappella-render" "$PLIST"
codesign --force --deep --sign - "$HELPER" >/dev/null 2>&1 || true
"$LSREGISTER" -f "$HELPER"

cat <<MSG

Installed:
  effect:    $VST
  extension: $EXT_DIR
  helper:    $HELPER

Next, in MuseScore 4:
  1. Quit and reopen MuseScore.
  2. On the Score page, the Extensions toolbar (top right) has a microphone
     button: Render, in one click, with the last settings. The cog next to it
     opens Settings and status. A notification says when a render is ready.
  3. Once per score: open the Mixer and add "Acappella Render" to the Master
     channel's Audio FX. Press Play: you hear the render, following the cursor.
     Switch the effect off to hear MuseScore's own sounds again.
MSG
if pgrep -qf "MuseScore 4.app/Contents/MacOS/mscore"; then
    echo "MuseScore is running: quit and reopen it. If the toolbar buttons don't show up,"
    echo "enable Acappella Render in Home > Plugins."
fi
