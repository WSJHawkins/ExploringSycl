#include "sycl_shared.hpp"
#include "../../shared.h"

using namespace cl::sycl;

// Copies the inner u into u0.
void copy_u(
        const int x, const int y, const int halo_depth, SyclBuffer& uBuff,
        SyclBuffer& u0Buff, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto u            = uBuff.get_access<access::mode::read>(cgh);
    auto u0           = u0Buff.get_access<access::mode::write>(cgh);

    //Define range
    auto myRange = range<1>(x*y);

    cgh.parallel_for<class copy_u>( myRange, [=] (id<1> idx){
      const size_t kk = idx[0] % x;
      const size_t jj = idx[0] / x;

      if(kk >= halo_depth && kk < x - halo_depth &&
         jj >= halo_depth && jj < y - halo_depth)
      {
          u0[idx[0]] = u[idx[0]];
      }
    });
  });//end of queue
}

// Calculates the residual r.
void calculate_residual(
            const int x, const int y, const int halo_depth, SyclBuffer& uBuff,
            SyclBuffer& u0Buff, SyclBuffer& rBuff, SyclBuffer& kxBuff,
            SyclBuffer& kyBuff, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto u            = uBuff.get_access<access::mode::read>(cgh);
    auto u0           = u0Buff.get_access<access::mode::read>(cgh);
    auto r            = rBuff.get_access<access::mode::write>(cgh);
    auto kx           = kxBuff.get_access<access::mode::read>(cgh);
    auto ky           = kyBuff.get_access<access::mode::read>(cgh);

    //Define range
    auto myRange = range<1>(x*y);

    cgh.parallel_for<class calculate_residual>( myRange, [=] (id<1> idx){
      const size_t kk = idx[0] % x;
      const size_t jj = idx[0] / x;

        if(kk >= halo_depth && kk < x - halo_depth &&
           jj >= halo_depth && jj < y - halo_depth)
        {
            //smvp uses kx and ky and INDEX and dims.x and dims.y!!!!
            int index = idx[0];
            const double smvp = SMVP(u);
            r[idx[0]] = u0[idx[0]] - smvp;
        }
    });
  });//end of queue
}

// Calculates the 2 norm of the provided buffer.
void calculate_2norm(
        const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
        double* norm, queue& device_queue)
{
  size_t wgroup_size = WORK_GROUP_SIZE; //define global
  auto len = bufferBuff.get_count();
  auto n_wgroups = floor((len + wgroup_size - 1) / wgroup_size);
  buffer<double, 1> tmpArrayBuff{range<1>{(size_t)n_wgroups}};

  device_queue.submit([&](handler &cgh) {
    auto buffer           = bufferBuff.get_access<access::mode::read>(cgh);
    auto global_mem       = tmpArrayBuff.get_access<access::mode::discard_write>(cgh);
    accessor <double, 1, access::mode::read_write, access::target::local> local_mem(range<1>(wgroup_size), cgh);

    auto myRange = nd_range<1>(n_wgroups * wgroup_size, wgroup_size);

    cgh.parallel_for<class calculate_2norm>( myRange, [=] (nd_item<1> item){
      size_t local_id = item.get_local_linear_id();
      size_t global_id = item.get_global_linear_id();
      local_mem[local_id] = 0;

      const size_t kk = global_id % x;
      const size_t jj = global_id / x;
      if(kk >= halo_depth && kk < x - halo_depth &&
         jj >= halo_depth && jj < y - halo_depth)
      {
          local_mem[local_id] = buffer[global_id]*buffer[global_id];
      }

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
});//end of queue
*norm += SyclHelper::reduceArray(tmpArrayBuff, device_queue);
}

// Finalises the energy field.
void finalise(
        const int x, const int y, const int halo_depth, SyclBuffer& uBuff,
        SyclBuffer& densityBuff, SyclBuffer& energyBuff, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto u            = uBuff.get_access<access::mode::read>(cgh);
    auto density      = densityBuff.get_access<access::mode::read>(cgh);
    auto energy       = energyBuff.get_access<access::mode::write>(cgh);

    //Define range
    auto myRange = range<1>(x*y);

    cgh.parallel_for<class finalise>( myRange, [=] (id<1> idx){
      const size_t kk = idx[0] % x;
      const size_t jj = idx[0] / x;
      if(kk >= halo_depth && kk < x - halo_depth &&
         jj >= halo_depth && jj < y - halo_depth)
      {
          energy[idx[0]] = u[idx[0]]/density[idx[0]];
      }
    });
  });//end of queue
}
