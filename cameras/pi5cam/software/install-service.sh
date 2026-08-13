#!/usr/bin/env bash
set -e

# Repo root, resolved relative to this script (cameras/pi5cam/software/ -> repo root)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd /opt/dusty
sudo cp "$REPO_ROOT/cameras/pi5cam/software/pi5cam/assets/dusty.service" /etc/systemd/system/dusty.service

sudo systemctl daemon-reload

sudo systemctl enable dusty.service
sudo systemctl start dusty.service
sudo systemctl status dusty.service
