# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

CFLAGS= -O3 -Wall
#-Wall -pedantic -ansi
#CFLAGS=-O4
SRCPATH=BiSUNA-U50/src
ENVPATH=${SRCPATH}/Environments/

MAIN_REINFORCEMENT_LEARNING=${SRCPATH}/host.cpp
ENVIRONMENTS=${ENVPATH}/ReinforcementEnvironment.cpp ${ENVPATH}/MountainCar.cpp
AGENTS=${SRCPATH}/RLAgent/UnifiedNeuralModel.cpp
NN=${SRCPATH}/NN/NNetwork.cpp ${SRCPATH}/NN/NNetworkModule.cpp ${SRCPATH}/NN/NNetworkState.cpp ${SRCPATH}/NN/NNoveltyMap.cpp ${SRCPATH}/NN/NNetworkExtra.cpp ${SRCPATH}/NN/NNetworkOCL.cpp
OCL=${SRCPATH}/OCL/OpenCLUtils.cpp ${SRCPATH}/OCL/OCLBridge.cpp ${SRCPATH}/OCL/OCLContainer.cpp
OTHER=${SRCPATH}/RandomUtils.cpp ${SRCPATH}/Configuration/PConfig.cpp

## Check pkg-config paths
## pkg-config --variable pc_path pkg-config

UNAME := $(shell uname)
LDFLAGS = -I${SRCPATH} -pthread

ifeq ($(UNAME), Linux)
	LDFLAGS += -lOpenCL -Wno-ignored-attributes -Wno-return-type -Wno-deprecated-declarations -Wno-unused-result -Wno-sign-compare
endif
ifeq ($(UNAME), Darwin)
	LDFLAGS += -framework OpenCL -Wno-missing-braces
endif

LDFLAGS += -L/usr/local/lib

#### Xilinx Vitis Section
#
# this section of the file was created by a computer, modified by a human. Do not trust it.
#

# compiler tools
XILINX_VITIS ?= /opt/xilinx/Vitis/2020.1
XILINX_XRT ?= /opt/xilinx/xrt
XILINX_VIVADO ?= /opt/xilinx/Vivado/2020.1
XILINX_VIVADO_HLS ?= $(XILINX_VITIS)/Vivado_HLS

HOST_CXX ?= g++
VPP ?= ${XILINX_VITIS}/bin/v++
RM = rm -f
RMDIR = rm -rf

VITIS_PLATFORM = xilinx_u50_gen3x16_xdma_201920_3
VITIS_PLATFORM_PATH = $(VITIS_PLATFORM)

# hardware compiler shared settings
VPP_OPTS = --target hw
#VPP_OPTS = --target hw_emu
#VPP_OPTS = --target sw_emu

#
# OpenCL kernel files
#

BUILD_SUBDIRS += BiGlobal.build
BIGLOBAL_OBJS += BiGlobal.build/processStateG.xo
ALL_KERNEL_OBJS += BiGlobal.build/processStateG.xo
BUILDVIVADO = BiGlobal.build/processStateG/

ALL_MESSAGE_FILES = $(subst .xo,.mdb,$(ALL_KERNEL_OBJS))

KERNEL = ${SRCPATH}/OCL/Kernels/ProcessStateGlobal.cl
COMPFLD = resources/ConfFPGA
COMPCFG = ${COMPFLD}/Compile-config.cfg
DCONT = -DCONTINUOUS_PARAM

#
# host files
#
# BUILD_SUBDIRS += ../src/OCL/Kernels/

#
# primary build targets
#

.PHONY: all clean
all: bisuna suna binary continuous

binary: BiGlobal-1.xclbin BiGlobal-2.xclbin BiGlobal-4.xclbin BiGlobal-8.xclbin BiGlobal-10.xclbin
continuous: CGlobal-1.xclbin CGlobal-2.xclbin CGlobal-4.xclbin CGlobal-8.xclbin CGlobal-10.xclbin

.NOTPARALLEL: clean

clean-accelerators:
	-$(RM) $(ALL_KERNEL_OBJS) $(ALL_MESSAGE_FILES)
	-$(RM) *.xclbin.sh *.xclbin.info *.xclbin.link_summary*
	-$(RMDIR) $(BUILD_SUBDIRS)
	-$(RMDIR) .Xil

clean-package:
	-${RMDIR} package
	-${RMDIR} package.build
	-${RM} xrc.log xcd.log *.ltx
	-${RM} bisuna suna
	-${RM} BiGlobal-1.xclbin BiGlobal-2.xclbin BiGlobal-4.xclbin BiGlobal-8.xclbin
	-${RM} CGlobal-1.xclbin CGlobal-2.xclbin CGlobal-4.xclbin CGlobal-8.xclbin

clean: clean-accelerators clean-package

.PHONY: incremental
incremental: all

nothing:

#
# Host code
#

bisuna:
	g++ -std=c++17 $(CFLAGS) $(PROFILE) $(MAIN_REINFORCEMENT_LEARNING) $(AGENTS) $(NN) $(ENVIRONMENTS) $(OCL) $(OTHER) $(LDFLAGS) -o bisuna

suna:
	g++ -std=c++17 -DCONTINUOUS_PARAM $(CFLAGS) $(PROFILE) $(MAIN_REINFORCEMENT_LEARNING) $(AGENTS) $(NN) $(ENVIRONMENTS) $(OCL) $(OTHER) $(LDFLAGS) -o suna

#
# FPGA BiGlobal.xclbin
#
BiGlobal.build/processStateG-1.xo: $(KERNEL) $(COMPCFG) ${COMPFLD}/${COMPFLD}/Con1.cfg
	-@$(RMDIR) $(BUILDVIVADO)
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" --config $(COMPCFG) --config ${COMPFLD}/Con1.cfg  -o"$@" "$<"

BiGlobal-1.xclbin: BiGlobal.build/processStateG-1.xo $(COMPCFG) ${COMPFLD}/Con1.cfg
	$(VPP) $(VPP_OPTS) --link --config $(COMPCFG) --config ${COMPFLD}/Con1.cfg  -o"$@" BiGlobal.build/processStateG-1.xo

BiGlobal.build/processStateG-2.xo: $(KERNEL) $(COMPCFG) ${COMPFLD}/Con2.cfg
	-@$(RMDIR) $(BUILDVIVADO)
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" --config $(COMPCFG) --config ${COMPFLD}/Con2.cfg  -o"$@" "$<"

BiGlobal-2.xclbin: BiGlobal.build/processStateG-2.xo $(COMPCFG) ${COMPFLD}/Con2.cfg
	$(VPP) $(VPP_OPTS) --link --config $(COMPCFG) --config ${COMPFLD}/Con2.cfg  -o"$@" BiGlobal.build/processStateG-2.xo

BiGlobal.build/processStateG-4.xo: $(KERNEL) $(COMPCFG) ${COMPFLD}/Con4.cfg
	-@$(RMDIR) $(BUILDVIVADO)
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" --config $(COMPCFG) --config ${COMPFLD}/Con4.cfg  -o"$@" "$<"

BiGlobal-4.xclbin: BiGlobal.build/processStateG-4.xo $(COMPCFG) ${COMPFLD}/Con4.cfg
	$(VPP) $(VPP_OPTS) --link --config $(COMPCFG) --config ${COMPFLD}/Con4.cfg  -o"$@" BiGlobal.build/processStateG-4.xo

BiGlobal.build/processStateG-8.xo: $(KERNEL) $(COMPCFG) ${COMPFLD}/Con8.cfg
	-@$(RMDIR) $(BUILDVIVADO)
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" --config $(COMPCFG) --config ${COMPFLD}/Con8.cfg  -o"$@" "$<"

BiGlobal-8.xclbin: BiGlobal.build/processStateG-8.xo $(COMPCFG) ${COMPFLD}/Con8.cfg
	$(VPP) $(VPP_OPTS) --link --config $(COMPCFG) --config ${COMPFLD}/Con8.cfg  -o"$@" BiGlobal.build/processStateG-8.xo

BiGlobal.build/processStateG-10.xo: $(KERNEL) $(COMPCFG) Con10.cfg
	-@$(RMDIR) $(BUILDVIVADO)
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" --config $(COMPCFG) --config Con10.cfg  -o"$@" "$<"

BiGlobal-10.xclbin: BiGlobal.build/processStateG-10.xo $(COMPCFG) Con10.cfg
	$(VPP) $(VPP_OPTS) --link --config $(COMPCFG) --config Con10.cfg  -o"$@" BiGlobal.build/processStateG-10.xo

####
#### Continous
####

BiGlobal.build/processStateGC-1.xo: $(KERNEL) $(COMPCFG) ${COMPFLD}/Con1.cfg
	-@$(RMDIR) $(BUILDVIVADO)
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" $(DCONT) --config $(COMPCFG) --config ${COMPFLD}/Con1.cfg  -o"$@" "$<"

CGlobal-1.xclbin: BiGlobal.build/processStateGC-1.xo $(COMPCFG) ${COMPFLD}/Con1.cfg
	$(VPP) $(VPP_OPTS) $(DCONT) --link --config $(COMPCFG) --config ${COMPFLD}/Con1.cfg  -o"$@" BiGlobal.build/processStateGC-1.xo

BiGlobal.build/processStateGC-2.xo: $(KERNEL) $(COMPCFG) ${COMPFLD}/Con2.cfg
	-@$(RMDIR) $(BUILDVIVADO)
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" $(DCONT) --config $(COMPCFG) --config ${COMPFLD}/Con2.cfg  -o"$@" "$<"

CGlobal-2.xclbin: BiGlobal.build/processStateGC-2.xo $(COMPCFG) ${COMPFLD}/Con2.cfg
	$(VPP) $(VPP_OPTS) $(DCONT) --link --config $(COMPCFG) --config ${COMPFLD}/Con2.cfg  -o"$@" BiGlobal.build/processStateGC-2.xo

BiGlobal.build/processStateGC-4.xo: $(KERNEL) $(COMPCFG) ${COMPFLD}/Con4.cfg
	-@$(RMDIR) $(BUILDVIVADO)
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" $(DCONT) --config $(COMPCFG) --config ${COMPFLD}/Con4.cfg  -o"$@" "$<"

CGlobal-4.xclbin: BiGlobal.build/processStateGC-4.xo $(COMPCFG) ${COMPFLD}/Con4.cfg
	$(VPP) $(VPP_OPTS) $(DCONT) --link --config $(COMPCFG) --config ${COMPFLD}/Con4.cfg  -o"$@" BiGlobal.build/processStateGC-4.xo

BiGlobal.build/processStateGC-8.xo: $(KERNEL) $(COMPCFG) ${COMPFLD}/Con8.cfg
	-@$(RMDIR) $(BUILDVIVADO)
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" $(DCONT) --config $(COMPCFG) --config ${COMPFLD}/Con8.cfg  -o"$@" "$<"

CGlobal-8.xclbin: BiGlobal.build/processStateGC-8.xo $(COMPCFG) ${COMPFLD}/Con8.cfg
	$(VPP) $(VPP_OPTS) $(DCONT) --link --config $(COMPCFG) --config ${COMPFLD}/Con8.cfg  -o"$@" BiGlobal.build/processStateGC-8.xo

BiGlobal.build/processStateGC-10.xo: $(KERNEL) $(COMPCFG) Con10.cfg
	-@$(RMDIR) $(BUILDVIVADO)
	-@mkdir -p $(@D)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" $(DCONT) --config $(COMPCFG) --config Con10.cfg  -o"$@" "$<"

CGlobal-10.xclbin: BiGlobal.build/processStateGC-10.xo $(COMPCFG) Con10.cfg
	$(VPP) $(VPP_OPTS) $(DCONT) --link --config $(COMPCFG) --config Con10.cfg  -o"$@" BiGlobal.build/processStateGC-10.xo
