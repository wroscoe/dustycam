.PHONY: build flash ota-deploy clean
IDF_IMG = espressif/idf:release-v5.5
ROOT = $(realpath ../..)
PORT ?= /dev/ttyACM0
DEV = xiaomic1

build:            ## build firmware in Docker
	date +v%Y%m%d-%H%M%S > version.txt
	docker run --rm -v $(ROOT):/work -w /work/firmware/miclogger $(IDF_IMG) \
	  bash -c "idf.py -DSDKCONFIG_DEFAULTS='sdkconfig.defaults;sdkconfig.secrets' build"

flash:            ## bootstrap flash over native USB-JTAG (only needed once)
	esptool --port $(PORT) --baud 921600 write-flash --flash-mode keep \
	  0x0 build/bootloader/bootloader.bin 0xc000 build/partition_table/partition-table.bin \
	  0x13000 build/ota_data_initial.bin 0x20000 build/miclogger.bin

ota-deploy: build ## publish to sensorhub; device updates within ~30s
	cp build/miclogger.bin ../../data/firmware/$(DEV).bin
	@echo "published $$(cat version.txt) for $(DEV)"

clean:
	docker run --rm -v $(ROOT):/work $(IDF_IMG) bash -c \
	  "rm -rf /work/firmware/miclogger/{build,sdkconfig,managed_components,dependencies.lock}"
