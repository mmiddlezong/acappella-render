-- Acappella Render Helper: handles acappella-render://render?score=<name>
-- links from the Render button on MuseScore's Extensions toolbar by starting
-- a render of that score in the background (the bridge checks the name
-- against MuseScore's recent files).  install.sh fills in the @...@ paths
-- and registers the link scheme.

property pythonPath : "@PYTHON@"
property bridgePath : "@BRIDGE@"
property logPath : "@LOG@"

on open location theURL
	if theURL does not start with "acappella-render://render?" then return
	do shell script quoted form of pythonPath & " " & quoted form of bridgePath & ¬
		" --url " & quoted form of theURL & " --notify >> " & quoted form of logPath & " 2>&1 &"
end open location

on run
end run
