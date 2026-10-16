// Acappella Render Helper: opens acappella-render://render?score=NAME links
// from the Render button on MuseScore's Extensions toolbar, starts the render
// (the bridge checks the name against MuseScore's recent files) and shows a
// small floating panel over MuseScore with its progress until it's ready.
// It never takes the keyboard from MuseScore.  install.sh fills in the
// @...@ paths, builds this into an app and registers the link scheme.
//
//   AcappellaRenderHelper --snapshot out.png working|ready|error TITLE DETAIL
// draws the panel into a PNG instead (for checking how it looks).

import Cocoa

let python = "@PYTHON@"
let bridge = "@BRIDGE@"
let dataDir = "@DATA@"

enum Look { case working, ready, error }

final class StatusPanel {
    let panel: NSPanel
    let background: NSVisualEffectView
    let title = NSTextField(labelWithString: "")
    let detail = NSTextField(labelWithString: "")
    let spinner = NSProgressIndicator()
    let badge = NSImageView()
    var onClick: (() -> Void)?

    init() {
        let size = NSSize(width: 380, height: 60)
        panel = NSPanel(contentRect: NSRect(origin: .zero, size: size),
                        styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.level = .floating
        panel.isFloatingPanel = true
        panel.hidesOnDeactivate = false
        panel.becomesKeyOnlyIfNeeded = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = true

        background = NSVisualEffectView(frame: NSRect(origin: .zero, size: size))
        background.material = .hudWindow
        background.blendingMode = .behindWindow
        background.state = .active
        background.wantsLayer = true
        background.layer?.cornerRadius = 12
        background.layer?.masksToBounds = true
        panel.contentView = background

        spinner.style = .spinning
        spinner.controlSize = .regular
        badge.imageScaling = .scaleProportionallyUpOrDown
        title.font = .boldSystemFont(ofSize: 13)
        title.lineBreakMode = .byTruncatingTail
        detail.font = .systemFont(ofSize: 11)
        detail.textColor = .secondaryLabelColor
        detail.cell?.wraps = true
        detail.cell?.truncatesLastVisibleLine = true
        layout(lines: 1)
        for view in [spinner, badge, title, detail] as [NSView] {
            background.addSubview(view)
        }
        background.addGestureRecognizer(NSClickGestureRecognizer(target: self, action: #selector(clicked)))
    }

    @objc func clicked() { onClick?() }

    // one line of detail, or two (an error message)
    func layout(lines: Int) {
        let h: CGFloat = lines > 1 ? 74 : 60
        var frame = panel.frame
        frame.origin.y += frame.height - h
        frame.size.height = h
        panel.setFrame(frame, display: false)
        background.frame = NSRect(x: 0, y: 0, width: frame.width, height: h)
        spinner.frame = NSRect(x: 14, y: h / 2 - 14, width: 28, height: 28)
        badge.frame = spinner.frame
        title.frame = NSRect(x: 52, y: h - 29, width: frame.width - 66, height: 18)
        detail.maximumNumberOfLines = lines
        detail.frame = NSRect(x: 52, y: 12, width: frame.width - 66, height: lines > 1 ? 30 : 16)
    }

    func set(_ look: Look, _ heading: String, _ text: String) {
        layout(lines: look == .error ? 2 : 1)
        title.stringValue = heading
        detail.stringValue = text
        switch look {
        case .working:
            badge.isHidden = true
            spinner.isHidden = false
            spinner.startAnimation(nil)
        case .ready, .error:
            spinner.stopAnimation(nil)
            spinner.isHidden = true
            badge.isHidden = false
            let name = look == .ready ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
            badge.image = NSImage(systemSymbolName: name, accessibilityDescription: nil)
            badge.contentTintColor = look == .ready ? .systemGreen : .systemOrange
        }
    }

    func show() {
        if !panel.isVisible, let screen = NSScreen.main {
            // top right, just under MuseScore's toolbars (where the button is)
            let f = screen.visibleFrame
            let size = panel.frame.size
            panel.setFrameOrigin(NSPoint(x: f.maxX - size.width - 18, y: f.maxY - size.height - 96))
        }
        panel.alphaValue = 1
        panel.orderFrontRegardless()
    }

    func hide(then done: @escaping () -> Void) {
        NSAnimationContext.runAnimationGroup({ ctx in
            ctx.duration = 0.4
            panel.animator().alphaValue = 0
        }, completionHandler: {
            self.panel.orderOut(nil)
            done()
        })
    }

    func snapshot(to path: String) {
        // the frosted background can't be captured off screen: draw a solid one
        background.material = .windowBackground
        background.blendingMode = .withinWindow
        background.layoutSubtreeIfNeeded()
        guard let rep = background.bitmapImageRepForCachingDisplay(in: background.bounds) else { return }
        background.cacheDisplay(in: background.bounds, to: rep)
        try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: path))
    }
}

final class Helper: NSObject, NSApplicationDelegate {
    let ui = StatusPanel()
    var request: String?
    var score = ""
    var started = Date()
    var finished: Date?
    var linger = 0.0
    var timer: Timer?

    func applicationWillFinishLaunching(_ notification: Notification) {
        NSAppleEventManager.shared().setEventHandler(
            self, andSelector: #selector(openLink(_:reply:)),
            forEventClass: AEEventClass(kInternetEventClass), andEventID: AEEventID(kAEGetURL))
        ui.onClick = { [weak self] in self?.dismiss() }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // opened without a link: nothing to do
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            if self.request == nil { NSApp.terminate(nil) }
        }
    }

    @objc func openLink(_ event: NSAppleEventDescriptor, reply: NSAppleEventDescriptor) {
        guard let link = event.paramDescriptor(forKeyword: keyDirectObject)?.stringValue,
              let parts = URLComponents(string: link), parts.scheme == "acappella-render",
              parts.host == "render",
              let name = parts.queryItems?.first(where: { $0.name == "score" })?.value
        else { return }
        start(link: link, name: name)
    }

    func start(link: String, name: String) {
        let id = "toolbar-\(Int(Date().timeIntervalSince1970 * 1000))"
        request = id
        score = name
        started = Date()
        finished = nil
        ui.set(.working, "Rendering \(name)", "Starting")
        ui.show()

        let logPath = dataDir + "/bridge.log"
        if !FileManager.default.fileExists(atPath: logPath) {
            FileManager.default.createFile(atPath: logPath, contents: nil)
        }
        let log = FileHandle(forWritingAtPath: logPath)
        log?.seekToEndOfFile()
        let p = Process()
        p.executableURL = URL(fileURLWithPath: python)
        p.arguments = [bridge, "--url", link, "--request", id]
        p.standardInput = FileHandle.nullDevice
        p.standardOutput = log
        p.standardError = log
        p.terminationHandler = { _ in
            DispatchQueue.main.async { self.bridgeExited(id) }
        }
        do {
            try p.run()
        } catch {
            finish(.error, "Couldn't start the renderer", error.localizedDescription, after: 12)
            return
        }
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in self.poll() }
    }

    func status() -> [String: Any]? {
        guard let data = FileManager.default.contents(atPath: dataDir + "/status.json") else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    func poll() {
        if let done = finished {
            if Date().timeIntervalSince(done) > linger { dismiss() }
            return
        }
        guard let id = request else { return }
        let elapsed = Int(Date().timeIntervalSince(started))
        var text = "Starting"
        if let s = status(), s["request"] as? String == id {
            let message = (s["message"] as? String) ?? ""
            switch s["state"] as? String {
            case "done":
                finish(.ready, "\(score) is ready. Press Play.", summary(message), after: 5)
                return
            case "error":
                finish(.error, "Couldn't render \(score)", message, after: 15)
                return
            default:
                text = message
            }
        }
        ui.set(.working, "Rendering \(score) · \(elapsed) s", text)
    }

    // "NAME: 157.2s of music, rendered in 4.1s, 7 of 9 voices unchanged"
    //   -> "Rendered in 4.1 s, 7 of 9 voices unchanged"
    func summary(_ message: String) -> String {
        guard let r = message.range(of: "rendered in ") else { return message }
        let rest = message[r.upperBound...].replacingOccurrences(
            of: #"^([0-9.]+)s"#, with: "$1 s", options: .regularExpression)
        return "Rendered in " + rest
    }

    func bridgeExited(_ id: String) {
        guard id == request, finished == nil else { return }
        poll()
        if finished == nil {
            finish(.error, "The renderer stopped", "See \(dataDir)/bridge.log", after: 15)
        }
    }

    func finish(_ look: Look, _ heading: String, _ text: String, after seconds: Double) {
        ui.set(look, heading, text)
        ui.show()
        finished = Date()
        linger = seconds
    }

    func dismiss() {
        if finished == nil { return }          // still rendering: keep showing it
        timer?.invalidate()
        request = nil
        ui.hide { NSApp.terminate(nil) }
    }
}

let args = CommandLine.arguments
if args.count >= 6 && args[1] == "--snapshot" {
    _ = NSApplication.shared
    let ui = StatusPanel()
    let look: Look = args[3] == "ready" ? .ready : args[3] == "error" ? .error : .working
    ui.set(look, args[4], args[5])
    ui.snapshot(to: args[2])
    exit(0)
}

let app = NSApplication.shared
let helper = Helper()
app.delegate = helper
app.setActivationPolicy(.accessory)
app.run()
