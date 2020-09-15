# Exploring the performance portability of the SYCL language

This repository contains my dissertation and all the code needed to replicate any of the findings. The instructions to run each program are found in their respective folders. 

I have ported an OpenCL version of a Lattice Boltzmann Code to SYCL. The original OpenCL and an OpenMP version are provided for comparison.

I have ported the kernels from TeaLeaf over to SYCL. I only provide the code to the SYCL Kernel. Other Kernels and other documentation for TeaLeaf can be found at https://github.com/UoB-HPC/TeaLeaf.

The 'How to port to SYCL' folder provides a guide (with examples) to port existing codes to the SYCL programming model. These guides are based on my experiences from undertaking this project and by no means cover all cases.

timings.sql contains a SQL table of all 466 timing results used in this project.
