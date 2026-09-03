.PHONY: build flash clean
IDF_IMG = espressif/idf:release-v5.5
ROOT = $(realpath ../..)
PORT ?= /dev/ttyUSB0

build:
	date +v%Y%m%d-%H%M%S > version.txt
	docker run --rm -v $(ROOT):/work -w /work/sensors/espnowbridge $(IDF_IMG) \
	  bash -c "idf.py -DSDKCONFIG_DEFAULTS='sdkconfig.defaults;sdkconfig.secrets' build"

flash: 
	esptool --port $(PORT) --baud 921600 write-flash --flash-mode keep \
	  0x0 build/bootloader/bootloader.bin 0xc000 build/partition_table/partition-table.bin 0x13000 build/ota_data_initial.bin \
	  0x20000 build/espnowbridge.bin

clean:
	docker run --rm -v $(ROOT):/work $(IDF_IMG) bash -c \
	  "rm -rf /work/sensors/espnowbridge/{build,sdkconfig}"

ota-deploy: build
	cp build/espnowbridge.bin /hd2/sensorhub/firmware/espnowbridge.bin
	@echo "published for espnowbridge"
