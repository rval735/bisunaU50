#
# this file was created by a computer. trust it.
#

# compiler tools
XILINX_VITIS ?= /opt/xilinx/Vitis/2020.1
XILINX_XRT ?= /opt/xilinx/xrt
XILINX_VIVADO ?= /opt/xilinx/Vivado/2020.1
XILINX_VIVADO_HLS ?= $(XILINX_VITIS)/Vivado_HLS

HOST_CXX ?= g++
VPP ?= ${XILINX_VITIS}/bin/v++
EMCONFIGUTIL = ${XILINX_VITIS}/bin/emconfigutil --od .
RM = rm -f
RMDIR = rm -rf

VITIS_PLATFORM = xilinx_u50_gen3x16_xdma_201920_3
VITIS_PLATFORM_PATH = $(VITIS_PLATFORM)

# host compiler global settings
CXXFLAGS += -std=c++1y -DVITIS_PLATFORM=$(VITIS_PLATFORM) -D__USE_XOPEN2K8 -I"/home/klugo/workspace/BiSUNA-U50/src" -I$(XILINX_XRT)/include/ -I$(XILINX_VIVADO)/include/ -O0 -g3 -p -Wall -c -fmessage-length=0 -DBREAKEARLY
LDFLAGS += -lxilinxopencl -lpthread -lrt -lstdc++ -lmpfr -lgmp -lhlsmc++-GCC46 -lIp_floating_point_v7_0_bitacc_cmodel -lIp_xfft_v9_1_bitacc_cmodel -lIp_fir_compiler_v7_2_bitacc_cmodel -lIp_dds_compiler_v6_0_bitacc_cmodel -L$(XILINX_XRT)/lib/ -L$(XILINX_VIVADO)/lnx64/tools/fpo_v7_0 -L$(XILINX_VIVADO)/lnx64/lib/csim -L$(XILINX_VIVADO)/lnx64/tools/dds_v6_0 -L$(XILINX_VIVADO)/lnx64/tools/fir_v7_0 -L$(XILINX_VIVADO)/lnx64/tools/fft_v9_1 -Wl,-rpath-link,$(XILINX_XRT)/lib -Wl,-rpath,$(XILINX_VIVADO)/lnx64/lib/csim -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/fpo_v7_0 -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/fft_v9_1 -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/fir_v7_0 -Wl,-rpath,$(XILINX_VIVADO)/lnx64/tools/dds_v6_0

# hardware compiler shared settings
VPP_OPTS = --target hw_emu

#
# OpenCL kernel files
#

BINARY_CONTAINERS += BiGlobal.xclbin

BUILD_SUBDIRS += BiGlobal.build
BIGLOBAL_OBJS += BiGlobal.build/processStateG.xo
ALL_KERNEL_OBJS += BiGlobal.build/processStateG.xo

ALL_MESSAGE_FILES = $(subst .xo,.mdb,$(ALL_KERNEL_OBJS)) $(subst .xclbin,.mdb,$(BINARY_CONTAINERS))

#
# host files
#

HOST_OBJECTS += src/Configuration/PConfig.o
HOST_OBJECTS += src/Environments/MountainCar.o
HOST_OBJECTS += src/Environments/ReinforcementEnvironment.o
HOST_OBJECTS += src/NN/NNetwork.o
HOST_OBJECTS += src/NN/NNetworkExtra.o
HOST_OBJECTS += src/NN/NNetworkModule.o
HOST_OBJECTS += src/NN/NNetworkOCL.o
HOST_OBJECTS += src/NN/NNetworkState.o
HOST_OBJECTS += src/NN/NNoveltyMap.o
HOST_OBJECTS += src/OCL/OCLBridge.o
HOST_OBJECTS += src/OCL/OCLContainer.o
HOST_OBJECTS += src/OCL/OpenCLUtils.o
HOST_OBJECTS += src/RLAgent/UnifiedNeuralModel.o
HOST_OBJECTS += src/RandomUtils.o
HOST_OBJECTS += src/host.o
HOST_OBJECTS += src/xcl2/xcl2.o

HOST_EXE = BiSUNA-U50

BUILD_SUBDIRS += src/Configuration/
BUILD_SUBDIRS += src/Environments/
BUILD_SUBDIRS += src/NN/
BUILD_SUBDIRS += src/OCL/
BUILD_SUBDIRS += src/RLAgent/
BUILD_SUBDIRS += src/
BUILD_SUBDIRS += src/xcl2/

EMCONFIG_FILE = emconfig.json

#
# primary build targets
#

.PHONY: all clean
all: $(BINARY_CONTAINERS) $(HOST_EXE) $(EMCONFIG_FILE)

.NOTPARALLEL: clean

clean-host:
	-$(RM) $(HOST_EXE) $(HOST_OBJECTS)

clean-accelerators:
	-$(RM) $(BINARY_CONTAINERS) $(ALL_KERNEL_OBJS) $(ALL_MESSAGE_FILES)  $(EMCONFIG_FILE)
	-$(RM) *.xclbin.sh *.xclbin.info *.xclbin.link_summary*
	-$(RMDIR) $(BUILD_SUBDIRS)
	-$(RMDIR) .Xil

clean-package:
	-${RMDIR} package
	-${RMDIR} package.build

clean: clean-host clean-accelerators clean-package

ifeq ($(MAKECMDGOALS),incremental)
.PHONY: BiGlobal.xclbin
.PHONY: BiGlobal.build/processStateG.xo
endif

.PHONY: incremental
incremental: all

include makeemconfig.mk

nothing:

#
# binary container: BiGlobal.xclbin
#

BiGlobal.build/processStateG.xo: ../src/OCL/Kernels/ProcessStateGlobal.cl BiGlobal-processStateG-compile.cfg common-config.cfg
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" --config common-config.cfg --config BiGlobal-processStateG-compile.cfg --profile_kernel data:all:all:all -o"$@" "$<"

BiGlobal.xclbin: $(BIGLOBAL_OBJS) BiGlobal-link.cfg common-config.cfg
	-@echo $(VPP) $(VPP_OPTS) --link --config common-config.cfg --config BiGlobal-link.cfg --profile_kernel data:all:all:all -o"$@" $(BIGLOBAL_OBJS) > BiGlobal.xclbin.sh
	$(VPP) $(VPP_OPTS) --link --config common-config.cfg --config BiGlobal-link.cfg --profile_kernel data:all:all:all -o"$@" $(BIGLOBAL_OBJS)

#
# host rules
#

src/Configuration/PConfig.o: ../src/Configuration/PConfig.cpp ../src/Configuration/PConfig.hpp ../src/Configuration/PConfDefaultValues.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/Environments/MountainCar.o: ../src/Environments/MountainCar.cpp ../src/Environments/MountainCar.hpp ../src/Environments/Discretizer.h ../src/RandomUtils.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/Environments/ReinforcementEnvironment.o: ../src/Environments/ReinforcementEnvironment.cpp ../src/Environments/ReinforcementEnvironment.hpp ../src/Environments/MountainCar.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/NN/NNetwork.o: ../src/NN/NNetwork.cpp ../src/NN/NNetwork.hpp ../src/RandomUtils.hpp ../src/Environments/ReinforcementEnvironment.hpp ../src/NN/NNetworkExtra.hpp ../src/OCL/OCLContainer.hpp ../src/NN/NNetworkOCL.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/NN/NNetworkExtra.o: ../src/NN/NNetworkExtra.cpp ../src/NN/NNetworkExtra.hpp ../src/Parameters.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/NN/NNetworkModule.o: ../src/NN/NNetworkModule.cpp ../src/NN/NNetworkModule.hpp ../src/RandomUtils.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/NN/NNetworkOCL.o: ../src/NN/NNetworkOCL.cpp ../src/NN/NNetworkOCL.hpp ../src/RandomUtils.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/NN/NNetworkState.o: ../src/NN/NNetworkState.cpp ../src/NN/NNetworkState.hpp ../src/RandomUtils.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/NN/NNoveltyMap.o: ../src/NN/NNoveltyMap.cpp ../src/NN/NNoveltyMap.hpp ../../../../../usr/include/assert.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/OCL/OCLBridge.o: ../src/OCL/OCLBridge.cpp ../src/OCL/OCLBridge.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/OCL/OCLContainer.o: ../src/OCL/OCLContainer.cpp ../src/OCL/OCLContainer.hpp ../src/RandomUtils.hpp ../src/OCL/Kernels/TycheI.h
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/OCL/OpenCLUtils.o: ../src/OCL/OpenCLUtils.cpp ../src/OCL/OpenCLUtils.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/RLAgent/UnifiedNeuralModel.o: ../src/RLAgent/UnifiedNeuralModel.cpp ../src/RLAgent/UnifiedNeuralModel.hpp ../src/NN/NNetworkModule.hpp ../src/RandomUtils.hpp ../src/NN/NNetworkExtra.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/RandomUtils.o: ../src/RandomUtils.cpp ../src/RandomUtils.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/host.o: ../src/host.cpp ../src/NN/NNetwork.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

src/xcl2/xcl2.o: ../src/xcl2/xcl2.cpp ../src/xcl2/xcl2.hpp
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

$(HOST_EXE): $(HOST_OBJECTS)
	$(HOST_CXX) -o "$@" $(+) $(LDFLAGS)
	-@echo $(VPP) --package --config package.cfg  BiGlobal.xclbin
	-@$(VPP) --package --config package.cfg  BiGlobal.xclbin


#
# emulation configuration file
#

$(EMCONFIG_FILE): makeemconfig.mk
	$(EMCONFIGUTIL) --nd $(NUMBER_OF_DEVICES)  --platform $(VITIS_PLATFORM_PATH)
	-@$(RMDIR) TempConfig .Xil/configutil*

