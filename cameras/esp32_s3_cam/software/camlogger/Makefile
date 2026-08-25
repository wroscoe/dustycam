.PHONY: build flash ota-deploy clean
IDF_IMG = espressif/idf:release-v5.5
ROOT = $(realpath ../..)

build:            ## build firmware in Docker
	date +v%Y%m%d-%H%M%S > version.txt
	docker run --rm -v $(ROOT):/work -w /work/firmware/camlogger $(IDF_IMG) \
	  bash -c "idf.py -DSDKCONFIG_DEFAULTS='sdkconfig.defaults;sdkconfig.secrets' build"

flash:            ## bootstrap flash over UART (only needed once)
	ESPTOOL_STUB_VERSION=2 esptool --port /dev/ttyUSB0 --baud 921600 write-flash \
	  0x0 build/bootloader/bootloader.bin 0xc000 build/partition_table/partition-table.bin \
	  0x13000 build/ota_data_initial.bin 0x20000 build/camlogger.bin

ota-deploy: build ## publish to sensorhub; device updates within ~30s
	cp build/camlogger.bin ../../data/firmware/goouuu1.bin
	@echo "published $$(cat version.txt) for goouuu1"

clean:
	docker run --rm -v $(ROOT):/work $(IDF_IMG) bash -c \
	  "rm -rf /work/firmware/camlogger/{build,sdkconfig,managed_components,dependencies.lock}"
