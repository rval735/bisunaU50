#
# This file was created by Xilinx Vitis tool.
#


#
# tools
#

RM = rm -f
RMDIR = rm -rf


XILINX_VITIS ?= /opt/xilinx/Vitis/2020.1

VPP ?= ${XILINX_VITIS}/bin/v++




VITIS_PLATFORM = xilinx_u50_gen3x16_xdma_201920_3
VITIS_PLATFORM_PATH = $(VITIS_PLATFORM)


#
# variables
#


#
# primary build targets
#

all: package

package:
	-@echo $(VPP) --package --config package.cfg  ../../BiSUNA-U50/Emulation-HW/BiGlobal.xclbin -o BiGlobal.xclbin
	-@$(VPP) --package --config package.cfg  ../../BiSUNA-U50/Emulation-HW/BiGlobal.xclbin -o BiGlobal.xclbin


clean:
	$(RMDIR) ./package.build
	${RMDIR} package

.PHONY: all package clean


