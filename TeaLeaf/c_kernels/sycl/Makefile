vpath %.cpp $(KERNELS_PATH)
vpath %.hpp $(KERNELS_PATH)

SYCL_COMPILER ?= LLVM
OptimisationLevel ?= O3

ifeq ($(SYCL_COMPILER), LLVM)
KERNEL_COMPILER = clang++
KERNEL_FLAGS	  = -$(OptimisationLevel) -std=c++11 -lOpenCL -fsycl --gcc-toolchain=/nfs/software/x86_64/gcc/7.4.0 -I$(KERNELS_PATH) $(OPTIONS)
TL_FLAGS	:= $(KERNEL_FLAGS)
endif

ifeq ($(SYCL_COMPILER), hipSYCL)
KERNEL_COMPILER = syclcc-clang
hip_Platform ?= rocm
ifeq ($(hip_Platform), cpu)
KERNEL_FLAGS	  = -std=c++17 -$(OptimisationLevel) --hipsycl-platform=$(hip_Platform) -I$(KERNELS_PATH) -fopenmp $(OPTIONS)
else
hip_Arch ?= gfx906
KERNEL_FLAGS	  = -std=c++17 -$(OptimisationLevel) --hipsycl-gpu-arch=$(hip_Arch) --hipsycl-platform=$(hip_Platform) -I$(KERNELS_PATH) $(OPTIONS)
endif
TL_FLAGS	:= $(KERNEL_FLAGS)
endif

ifeq ($(SYCL_COMPILER), computeCPP)
KERNEL_COMPILER = $(CXX)
COMPUTECPP_FLAGS = $(shell $(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/computecpp_info --dump-device-compiler-flags)
KERNEL_FLAGS	  = -I$(KERNELS_PATH) -$(OptimisationLevel) -std=c++11 -DSYCL -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include $(OPTIONS)
TL_FLAGS	:= $(KERNEL_FLAGS) -L$(COMPUTECPP_PACKAGE_ROOT_DIR)/lib -lComputeCpp -lOpenCL -Wl,--rpath=$(COMPUTECPP_PACKAGE_ROOT_DIR)/lib/
endif


# Override the default compiler and flags
TL_LINK	    := $(KERNEL_COMPILER)
TL_COMPILER := $(KERNEL_COMPILER)
TL_LDFLAGS 	+=  -Lobj/$(KERNELS)

# Locate the kernel files to compile
KERNEL_SRCS := $(wildcard $(KERNELS_PATH)/*.cpp)
KERNEL_OBJS := $(patsubst $(KERNELS_PATH)/%.cpp, obj/$(KERNELS)/%.o, $(KERNEL_SRCS))

build_kernels: $(KERNEL_OBJS)
	-@mv *.o *.a obj/$(KERNELS)/
	@echo "Built Sycl kernels"

ifeq ($(SYCL_COMPILER), computeCPP)
obj/$(KERNELS)/%.sycl: %.cpp Makefile make.deps
	$(COMPUTECPP_PACKAGE_ROOT_DIR)/bin/compute++ -DSYCL $(COMPUTECPP_FLAGS) -c -I$(KERNELS_PATH) -I$(COMPUTECPP_PACKAGE_ROOT_DIR)/include $< -o $@

obj/$(KERNELS)/%.o: %.cpp obj/$(KERNELS)/%.sycl Makefile make.deps
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) -include $(word 2,$^)  -c  $< -o $@

else
obj/$(KERNELS)/%.o: %.cpp Makefile make.deps
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) -c $< -o $@
endif

# Have to build the chunk independently
obj/$(KERNELS)/chunk.o: chunk.c Makefile make.deps
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) -c $< -o $@
