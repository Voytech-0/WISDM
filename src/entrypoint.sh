#!/bin/sh
# Start the Flask health endpoint in the background and then exec the container CMD.
# The health app is at src/health.py and will listen on port 8000.

# Start the Flask app in the background
python src/health.py &

# If no command was provided, keep the container alive so the health endpoint stays up
if [ "$#" -eq 0 ]; then
  tail -f /dev/null
else
  # Try to run the provided command; if it fails, keep container alive so health endpoint stays up for debugging
  "$@" || tail -f /dev/null
fi
