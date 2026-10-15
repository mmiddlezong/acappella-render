// Acappella Render for MuseScore 4: renders the open score with the
// a cappella renderer, then MuseScore's own playback (play button, cursor,
// loops, starting from a selected note) plays the render instead of
// MuseScore's sounds, through the "Acappella Render" effect on the Master
// channel.  install.sh fills in the @...@ paths.

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import MuseScore 3.0
import FileIO 3.0
import Muse.UiComponents

MuseScore {
    id: plugin
    title: "Acappella Render"
    description: "Render the score with acappella-render and hear it through MuseScore's playback"
    version: "1.0"
    categoryCode: "playback"
    pluginType: "dialog"
    requiresScore: true

    width: 440
    height: 470

    readonly property string python: "@PYTHON@"
    readonly property string bridge: "@BRIDGE@"
    readonly property string dataDir: "@DATA@"

    property string request: ""
    property string phase: ""        // "", rendering, done, error
    property string message: ""
    property string notes: ""
    property real startedAt: 0
    property int elapsed: 0

    Settings {
        category: "AcappellaRender"
        property alias singers: singersCtl.currentValue
        property alias neutral: neutralBox.checked
        property alias underneath: underneathBox.checked
    }

    QProcess { id: proc }
    FileIO { id: statusFile; source: plugin.dataDir + "/status.json" }

    Timer {
        interval: 400
        running: true
        repeat: true
        onTriggered: plugin.poll()
    }

    onRun: poll()

    function poll() {
        if (phase === "rendering") {
            elapsed = Math.round((Date.now() - startedAt) / 1000)
        }
        if (!statusFile.exists()) {
            return
        }
        var s
        try {
            s = JSON.parse(statusFile.read())
        } catch (e) {
            return          // being rewritten
        }
        // a render this dialog started, or (when just opened) the last one
        if (request !== "" && s.request !== request) {
            return
        }
        if (s.state === "rendering" && Date.now() - s.started * 1000 > 15 * 60 * 1000) {
            s.state = "error"
            s.message = "The last render never finished (see " + dataDir + "/bridge.log)."
        }
        if (s.state === "rendering") {
            if (phase !== "rendering") {
                startedAt = s.started * 1000
            }
        } else if (s.state === "done") {
            var when = new Date(s.finished * 1000)
            message = s.message + "\nReady " + when.toLocaleTimeString(Qt.locale(), "HH:mm") + ". Press Play in MuseScore."
        } else {
            message = s.message
        }
        phase = s.state
        notes = (s.notes || []).join("\n")
    }

    function shellArgs(args) {
        // run in the background so the render outlives this dialog
        return ["-c", "\"$@\" >>\"$0\" 2>&1 </dev/null &", dataDir + "/bridge.log"].concat(args)
    }

    function museScoreGain() {
        return underneathBox.checked ? "0.3" : "0"
    }

    function render() {
        if (phase === "rendering") {
            return
        }
        cmd("file-save")            // the renderer reads the saved .mscz
        request = String(Date.now())
        phase = "rendering"
        message = "Rendering " + curScore.scoreName + "..."
        notes = ""
        startedAt = Date.now()
        elapsed = 0
        proc.startWithArgs("/bin/sh", shellArgs([python, bridge,
            "--score-name", curScore.scoreName, "--request", request,
            "--singers", String(singersCtl.currentValue),
            "--neutral", neutralBox.checked ? "ah" : "",
            "--musescore-gain", museScoreGain()]))
        proc.waitForFinished(3000)
    }

    function setMix() {
        proc.startWithArgs("/bin/sh", shellArgs([python, bridge, "--mix-only",
            "--musescore-gain", museScoreGain()]))
        proc.waitForFinished(3000)
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 12

        StyledTextLabel {
            Layout.fillWidth: true
            horizontalAlignment: Text.AlignLeft
            wrapMode: Text.WordWrap
            text: "Renders this score with the a cappella renderer. MuseScore's play button, "
                  + "cursor and loops then play the render instead of MuseScore's sounds."
        }

        RowLayout {
            spacing: 12

            FlatButton {
                text: plugin.phase === "rendering" ? "Rendering... " + plugin.elapsed + " s" : "Render"
                accentButton: true
                enabled: plugin.phase !== "rendering"
                onClicked: plugin.render()
            }

            StyledTextLabel {
                text: "Saves the score first."
            }
        }

        StyledTextLabel {
            Layout.fillWidth: true
            visible: plugin.message !== ""
            horizontalAlignment: Text.AlignLeft
            wrapMode: Text.WordWrap
            font: ui.theme.bodyBoldFont
            color: plugin.phase === "error" ? "#d0453a" : ui.theme.fontPrimaryColor
            text: plugin.message
        }

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: plugin.notes !== ""
            clip: true

            StyledTextLabel {
                width: parent.width
                horizontalAlignment: Text.AlignLeft
                verticalAlignment: Text.AlignTop
                wrapMode: Text.WordWrap
                maximumLineCount: 200
                text: plugin.notes
            }
        }

        Item {
            Layout.fillHeight: true
            visible: plugin.notes === ""
        }

        SeparatorLine {}

        RowLayout {
            spacing: 8

            StyledTextLabel {
                text: "Singers per backing part"
            }

            IncrementalPropertyControl {
                id: singersCtl
                implicitWidth: 60
                currentValue: 2
                minValue: 1
                maxValue: 4
                step: 1
                decimals: 0
                onValueEdited: function(newValue) { currentValue = newValue }
            }
        }

        CheckBox {
            id: neutralBox
            text: "Sing lyric words on \"ah\""
            onClicked: checked = !checked
        }

        CheckBox {
            id: underneathBox
            text: "Keep MuseScore's own sound underneath, quietly"
            onClicked: {
                checked = !checked
                plugin.setMix()
            }
        }

        StyledTextLabel {
            Layout.fillWidth: true
            horizontalAlignment: Text.AlignLeft
            wrapMode: Text.WordWrap
            opacity: 0.75
            text: "Once per score: open the Mixer and add \"Acappella Render\" to the Master "
                  + "channel's Audio FX. Switch the effect off to hear MuseScore's sounds again."
        }
    }
}
