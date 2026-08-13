PORT ?= $(shell python3 tools/findport.py)
MP = PORT=$(PORT) tools/mp

.PHONY: setup test deploy run dev repl reset ls

setup:            ## create venv + install tools
	uv venv --python 3.12 && uv pip install mpremote pytest

test:             ## host-side unit tests (no board needed)
	.venv/bin/pytest tests/ -q

SRC = src/boardcam.py src/brightness.py src/app.py src/storage.py src/uplink.py src/logger_app.py

secrets:          ## regenerate src/secrets.py from ~/.dusty/ (source of truth)
	@dusty generate esp32_s3_cam 2>/dev/null \
	  || echo "note: \`dusty\` not on PATH - using existing src/secrets.py"

deploy: secrets   ## copy src/ modules to board flash (NOT main.py)
	$(MP) cp $(SRC) :
	@test -f src/secrets.py && PORT=$(PORT) tools/mp cp src/secrets.py : || echo "note: src/secrets.py missing - WiFi upload disabled"

deploy-autostart: deploy  ## also deploy main.py: board runs the logger on every boot!
	$(MP) cp src/main.py :

logger:           ## run the logger interactively (20 captures, output visible)
	$(MP) reset
	sleep 4
	.venv/bin/python tools/run_app.py "import logger_app; logger_app.run(count=20, echo=True)"

server:           ## start the image collection server (Ctrl-C to stop)
	python3 tools/server.py 8077

ota-deploy:       ## build detector firmware in Docker and publish for OTA
	date +v%Y%m%d-%H%M%S > persondet_app/version.txt
	docker run --rm -v $(CURDIR):/work -w /work/persondet_app \
	  espressif/idf:release-v5.5 bash -c \
	  "idf.py -DSDKCONFIG_DEFAULTS='sdkconfig.defaults;sdkconfig.secrets' build"
	mkdir -p tools/firmware
	cp persondet_app/build/person_detection.bin tools/firmware/
	@echo "published $$(cat persondet_app/version.txt) — board updates within ~30s"

run: deploy       ## hard-reset (clears any wedged camera), then run the demo
	$(MP) reset
	sleep 4
	.venv/bin/python tools/run_app.py 10

dev: run          ## edit src/, then `make dev` again — deploy+run is the loop

repl:             ## plain REPL
	$(MP) repl

reset:            ## hard-reset the board
	$(MP) reset

ls:               ## list files on the board
	$(MP) fs ls

# NOTE: never use `mpremote mount` on this board — mounted imports ride the
# flaky CDC link and a mid-import stall wedges the interpreter with the
# camera initialized (replug-only recovery). Code goes on flash, always.

autolabel:        ## label dataset/samples with YOLO teacher (containerized)
	docker run --rm -v $(CURDIR):/work -w /work python:3.12-slim bash -c \
	  "apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null && pip install -q --no-cache-dir ultralytics --extra-index-url https://download.pytorch.org/whl/cpu && python tools/autolabel.py"

train:            ## fine-tune person model on labeled samples (containerized)
	docker run --rm -v $(CURDIR):/work -w /work python:3.12-slim bash -c \
	  "pip install -q --no-cache-dir tensorflow-cpu pillow && python tools/train.py"
