COMPILER ?= LLVM
TARGET = d2q9-bgk
CXX ?= g++

ifeq ($(COMPILER), LLVM)
CC = clang++
CC_FLAGS	  = -O3 -std=c++11 -lOpenCL -fsycl --gcc-toolchain=/nfs/software/x86_64/gcc/7.4.0
endif

ifeq ($(COMPILER), hipSYCL)
CC = syclcc-clang
hip_Platform ?= rocm
ifeq ($(hip_Platform), cpu)
CC_FLAGS	  = -std=c++17 -O3 --hipsycl-platform=$(hip_Platform) -I$(KERNELS_PATH)
else
hip_Arch ?= gfx906
CC_FLAGS	  = -std=c++17 -O3 --hipsycl-gpu-arch=$(hip_Arch) --hipsycl-platform=$(hip_Platform) -I$(KERNELS_PATH)
endif
endif

ifeq ($(COMPILER), computeCPP)
COMPUTECPP_FLAGS = $(shell $(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/computecpp_info --dump-device-compiler-flags)
endif

all: $(TARGET)

ifeq ($(COMPILER), computeCPP)
$(TARGET):  $(TARGET).o $(TARGET).sycl
	$(CXX) -O3 -std=c++11 -DSYCL $(TARGET).o -L$(COMPUTECPP_PACKAGE_ROOT_DIR)/lib -lComputeCpp -lOpenCL -Wl,--rpath=$(COMPUTECPP_PACKAGE_ROOT_DIR)/lib/ -o $(TARGET)

$(TARGET).o: $(TARGET).cpp $(TARGET).sycl
	$(CXX) -O3 -std=c++11 -DSYCL $(TARGET).cpp -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -include $(TARGET).sycl $(EXTRA_FLAGS) -o $(TARGET).o

$(TARGET).sycl: $(TARGET).cpp
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ -DSYCL $(TARGET).cpp $(COMPUTECPP_FLAGS) -c -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include -o $(TARGET).sycl
else

$(TARGET): $(TARGET).cpp
	$(CC) $(CC_FLAGS) $(TARGET).cpp -o $(TARGET)

endif

CheckSize?=128x128
FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=../check/$(CheckSize).final_state.dat
REF_AV_VELS_FILE=../check/$(CheckSize).av_vels.dat

check:
	python ../check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(TARGET) av_vels.dat final_state.dat d2q9-bgk.sycl d2q9-bgk.o
