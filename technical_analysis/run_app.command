#!/bin/bash
# Double-click in Finder (or run `bash run_app.command`) to install the
# requirements into .venv, run the test suite and open the app in your browser.
# Close this window or press Ctrl+C to stop the app.
cd "$(dirname "$0")" || exit 1
set -o pipefail
mkdir -p .tmp
LOG=.tmp/run_app.log
APP_LOG=.tmp/streamlit.log
: > "$LOG"
say() { echo "$*" | tee -a "$LOG"; }
pause_and_exit() { read -r -p "Press Return to close."; exit "$1"; }

PY=""
for candidate in python3.12 python3.13 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    PY="$candidate"; break
  fi
done
if [ -z "$PY" ]; then
  say "Python 3.11 or newer was not found. Install it from python.org, then run this again."
  pause_and_exit 1
fi

if [ ! -x .venv/bin/python ]; then
  say "Creating .venv with $("$PY" --version 2>&1)"
  "$PY" -m venv .venv >>"$LOG" 2>&1 || { say "Could not create .venv (details in $LOG)."; pause_and_exit 1; }
fi
say "Python: $(.venv/bin/python --version 2>&1)"

say "Installing requirements (the first run takes a minute or two)..."
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements-dev.txt 2>&1 | tee -a "$LOG" ||
  { say "Installing requirements failed (details in $LOG)."; pause_and_exit 1; }
say "Streamlit: $(.venv/bin/python -m streamlit version 2>&1 | tail -1)"

say "Running tests..."
if .venv/bin/python -m pytest -q -p no:cacheprovider --disable-warnings 2>&1 | tee -a "$LOG"; then
  say "Tests passed."
else
  say "Some tests failed (details in $LOG). Starting the app anyway."
fi

say "Starting the app..."
: > "$APP_LOG"
.venv/bin/python -m streamlit run app.py >>"$APP_LOG" 2>&1 &
APP=$!
trap 'kill "$APP" 2>/dev/null' EXIT INT TERM HUP
URL=""
for _ in $(seq 120); do
  URL=$(grep -Eo 'http://(127\.0\.0\.1|localhost):[0-9]+' "$APP_LOG" | head -1)
  [ -n "$URL" ] && break
  kill -0 "$APP" 2>/dev/null || break
  sleep 0.5
done
if [ -z "$URL" ]; then
  say "The app did not start. Its output:"; cat "$APP_LOG" | tee -a "$LOG"
  pause_and_exit 1
fi
say "App running at $URL"
open "$URL"
say "Keep this window open while you use the app. Close it or press Ctrl+C to stop."
wait "$APP"
say "The app stopped (output in $APP_LOG)."
