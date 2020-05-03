# Lattice Boltzmann Code

This directory contains all my Lattice Boltzmann Code. The OpenCL and OpenMP versions were written during COMS30006 a unit in High Performance Computing at the University of Bristol.

## Compilation
Each folder contains a Makefile which will compile the code. Each has a default compiler set but this can be overridden by passing a ```COMPILER=``` variable to the make command.

For OpenCL and OpenMP you can choose between gcc and icc by passing ```COMPILER=gcc``` and ```COMPILER=icc``` respectively.

For SYCL you have three choices. Intel's LLVM, Codeplay's ComputeCPP and hipSYCL. Enter ```COMPILER=LLVM``` for Intel's LLVM SYCL Compiler; Enter ```COMPILER=computeCPP``` for Codeplay's ComputeCPP compiler and finally enter ```COMPILER=hipSYCL``` to use hipSYCL. When using hipSYCL also pass the following arguements to make ```hip_Arch = gfx906 hip_Platform = rocm``` to specify your architecture and platform

## Running
All the makefiles will produce an output file to run called ```d2q9-bgk```. To run this, enter the following command:
```./d2q9-bgk ../Inputs/input_128x128.params ../Obstacles_1024x1024.dat```
Change out ```128x128``` for other input sizes as applicable. The following sizes are provided: ```128x128```,```128x256```,```256x256```,```1024x1024```,```2048x2048```,```4096x4096```. 

When run the program will produce two files: ```av_vels.dat``` and ```final_state.dat```. For the ```1024x1024``` size and below, this output can be checked automatically. This is done by typing ```make check CheckSize=128x128``` replacing the size parameter where necessary.
