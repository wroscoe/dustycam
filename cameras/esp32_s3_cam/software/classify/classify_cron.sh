#!/bin/bash
# Hourly label refresh for the pump listener UI. In the midnight run, also
# finish yesterday (its last events would otherwise never be classified).
set -e
cd "$(dirname "$0")"
if [ "$(date +%H)" = "00" ]; then
  .venv/bin/python classify_day.py "$(date -d yesterday +%F)"
  [ -f /hd2/pumpaudio/fingerprint.json ] && \
    .venv/bin/python fingerprint.py scan "$(date -d yesterday +%F)"
fi
.venv/bin/python classify_day.py
[ -f /hd2/pumpaudio/fingerprint.json ] && .venv/bin/python fingerprint.py scan
exit 0
