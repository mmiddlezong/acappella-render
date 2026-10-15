// The one-click Render button on MuseScore's Extensions toolbar: saves the
// score and has it rendered in the background with the settings of the last
// render (set in "Settings and status").  Voices that didn't change are
// reused, and a notification says when it's ready.
//
// Extension scripts can't start programs, but they can open URLs: the
// acappella-render:// link goes to the small helper app install.sh sets up,
// which starts the render.

function main() {
    if (!curScore) {
        return;
    }
    cmd("file-save");               // the renderer reads the saved .mscz
    var interactive = require("MuseApi.Interactive");
    interactive.openUrl("acappella-render://render?score=" + encodeURIComponent(curScore.scoreName));
}
