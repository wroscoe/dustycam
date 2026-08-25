#!/bin/bash
# Stage board/logger.py as the plant sensor's OTA firmware on sensorhub.
# The board polls /firmware/plant/version each hourly wake and self-updates.
set -e
cd "$(dirname "$0")"

python3 -m py_compile board/logger.py     # never publish a syntax error

HASH=$(sha256sum board/logger.py | cut -c1-8)
VER="$(date +%Y%m%d-%H%M)-$HASH"

install -m 644 board/logger.py /hd2/sensorhub/firmware/plant.py
echo "$VER" > /hd2/sensorhub/firmware/plant.version

echo "staged plant.py as version $VER"
echo "board will pick it up on its next wake (<=1h, or ~10min on USB retry cadence)"
