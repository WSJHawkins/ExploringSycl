# Porting a program to SYCL

This README aims to help future developers put programs to the SYCL Programming lanaguage. It uses the knowledge I have gain from my experience of porting a Lattice Boltzmann Code from OpenCL to SYCL and from porting the TeaLeaf mini-app from Kokkos to SYCL. Although the guide focusses on porting existing codes from OpenCL or Kokkos much of it will be useful for anyone writing a program in SYCL. A great resource for learning SYCL is https://github.com/codeplaysoftware/syclacademy

## Starting from OpenCL


## Starting from Kokkos
Starting from Kokkos makes the task slightly easier as the code should already be in C++. All these code examples are adapted from the TeaLeaf kernel code.

Step 1. Change over all the parallel for constructs from Kokkos to SYCL.
```
//Kokkos Parallel For
parallel_for(x*y, KOKKOS_LAMBDA (const int index)
{
    u(index) = energy(index)*density(index);
});

//SYCL Parallel For    
device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto u       = uBuff.get_access<access::mode::write>(cgh);
    auto density = densityBuff.get_access<access::mode::read>(cgh);
    auto energy  = energyBuff.get_access<access::mode::read>(cgh);
    
    auto myRange = range<1>(x*y);
    cgh.parallel_for<class example>( myRange, [=] (id<1> idx){
    
        u[idx[0]] = energy[idx[0]]*density[idx[0]];
    });
    
});//end of queue
```

Step 2. Change over all the parallel reduce constructs from Kokkos to SYCL. The reduce array function in this example can be found it the code samples above. 

```
//Kokkos Reduction
parallel_reduce(x*y, KOKKOS_LAMBDA (const int index, double& norm_temp){
    norm_temp += buffer(index)*buffer(index);
}, *norm);

//SYCL Reduction
buffer<double, 1> tmpArrayBuff{range<1>{(size_t)x*y}};
device_queue.submit([&](handler &cgh) {
    auto buffer   = bufferBuff.get_access<access::mode::read>(cgh);
    auto tmpArray = tmpArrayBuff.get_access<access::mode::discard_write>(cgh);

    auto myRange = range<1>(x*y);
    cgh.parallel_for<class reduction_example>( myRange, [=] (id<1> idx){
      tmpArray[idx[0]] = 0;
        tmpArray[idx[0]] = buffer[idx[0]]*buffer[idx[0]];
    });//end of parallel for
    
}); //end of queue
*norm += SyclHelper::reduceArray(tmpArrayBuff, device_queue)

```
This solution will work but may not be optimal. A more optimal solution would be to do the reduction in place. This means instead of loading the results straight into the tmpArrayBuffer you reduce the elements first. This is shown below:

```
//SYCL Reduction In place
size_t wgroup_size = WORK_GROUP_SIZE;
  auto len = wBuff.get_count();
  auto n_wgroups = floor((len + wgroup_size - 1) / wgroup_size);
  buffer<double, 1> tmpArrayBuff{range<1>{(size_t)n_wgroups}};

  device_queue.submit([&](handler &cgh) {
    auto buffer   = bufferBuff.get_access<access::mode::read>(cgh);
    auto global_mem = tmpArrayBuff.get_access<access::mode::discard_write>(cgh);
    accessor <double, 1, access::mode::read_write, access::target::local> local_mem(range<1>(wgroup_size), cgh);

    auto myRange = nd_range<1>(n_wgroups * wgroup_size, wgroup_size);
    cgh.parallel_for<class cg_calc_w>( myRange, [=] (nd_item<1> item){

      size_t local_id = item.get_local_linear_id();
      size_t global_id = item.get_global_linear_id();
      
      local_mem[local_id] =  buffer[global_id]*buffer[global_id];

      item.barrier(access::fence_space::local_space);

      for (size_t stride = 1; stride < wgroup_size; stride *= 2) {
        auto idx = 2 * stride * local_id;
        if (idx < wgroup_size) {
          local_mem[idx] = local_mem[idx] + local_mem[idx + stride];
        }
        item.barrier(access::fence_space::local_space);
      }

      if (local_id == 0) {
        global_mem[item.get_group_linear_id()] = local_mem[0];
      }

    });//end of parallel for
      
}); //end of queue
*norm += SyclHelper::reduceArray(tmpArrayBuff, device_queue)
    
```

Step 3. Change all of the Kokkos View constructs over to Sycl Buffer constructs. Be careful here with pointers to make sure the data is held correctly.

KVIEW sycl buffer

Step 4. Find all deep copies. If it is from host to device you do not need to do anything as long as your kernels have accessors set up. If the copy is from device to host, find where the data in the host mirror is used and replace it with a host accessor as shown below. Now remove all host mirrors and deep copies.\\
```
//Deep Copy - copies from 2nd arg to 1st
Kokkos::deep_copy(host_mirror, buffer);

//If a copy to host use this to access
auto hostAcc = buffer.get_access<access::mode::read>();
```
