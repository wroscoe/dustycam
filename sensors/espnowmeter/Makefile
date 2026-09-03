.PHONY: build flash clean
IDF_IMG = espressif/idf:release-v5.5
ROOT = $(realpath ../..)
PORT ?= /dev/ttyACM0

build:
	docker run --rm -v $(ROOT):/work -w /work/sensors/espnowmeter $(IDF_IMG) \
	  bash -c "idf.py build"

flash:
	esptool --port $(PORT) --baud 921600 write-flash --flash-mode keep \
	  0x0 build/bootloader/bootloader.bin 0x8000 build/partition_table/partition-table.bin \
	  0x10000 build/espnowmeter.bin

clean:
	docker run --rm -v $(ROOT):/work $(IDF_IMG) bash -c \
	  "rm -rf /work/sensors/espnowmeter/{build,sdkconfig}"
