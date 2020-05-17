# TeaLeaf Code

This directory contains all my TeaLeaf Code. This repository only contains a SYCL kernel. For other kernels go to https://github.com/UoB-HPC/TeaLeaf.

## Compilation
For the SYCL Kernels use the following make command specifying the SYCL compiler you want to use:
```make KERNELS=sycl OPTIONS=-DNO_MPI SYCL_COMPILER= ```
Enter ```LLVM``` for Intel's LLVM SYCL Compiler; Enter ```computeCPP``` for Codeplay's ComputeCPP compiler and finally enter ```hipSYCL``` to use hipSYCL. When using hipSYCL also pass the following arguements to make ```hip_Arch = gfx906 hip_Platform = rocm``` to specify your architecture and platform.
The optimisation level is set at ```O3``` but if you wish to override this just enter ```OptimisationLevel=O3``` changing out ```O3``` for your desired level.

## To Run
To run the code enter ```./tealeaf```. It will run based on the parameters in the file ```tea.in```. For any more information on TeaLeaf check out https://github.com/UoB-HPC/TeaLeaf

## Benchmarks
The benchmarks provided in this repository have been modified from the originals found at: https://github.com/UK-MAC/TeaLeaf_ref. This offers no change to the functionality but instead is a change of variable names to keep them inline with how the C based host code parses the input file. The host code in this repository has also been changed to default to the C Kernels when nothing is specified.

## Profiling 
The code comes with a build in profiler for the kernels. To use this, add ```-DENABLE_PROFILING``` to the ```OPTIONS``` param specified to the make command.
