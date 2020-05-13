#include "sycl_shared.hpp"
#include "../../shared.h"

using namespace cl::sycl;

// Initialises p,r,u,w
void cg_init_u(
            const int x, const int y, const int coefficient,
            SyclBuffer& pBuff, SyclBuffer& rBuff, SyclBuffer& uBuff, SyclBuffer& wBuff, SyclBuffer& densityBuff,
            SyclBuffer& energyBuff, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto p            = pBuff.get_access<access::mode::discard_write>(cgh);
    auto r            = rBuff.get_access<access::mode::discard_write>(cgh);
    auto u            = uBuff.get_access<access::mode::discard_write>(cgh);
    auto w            = wBuff.get_access<access::mode::discard_write>(cgh);
    auto density      = densityBuff.get_access<access::mode::read>(cgh);
    auto energy       = energyBuff.get_access<access::mode::read>(cgh);

    //Define range
    auto myRange = range<1>(x*y);

    cgh.parallel_for<class cg_init_u>( myRange, [=] (id<1> idx){
      const size_t kk = idx[0] % x;
      const size_t jj = idx[0] / x;

      p[idx[0]] = 0.0;
      r[idx[0]] = 0.0;
      u[idx[0]] = energy[idx[0]]*density[idx[0]];
      if(jj > 0 && jj < y-1 && kk > 0 & kk < x-1) {
        w[idx[0]] = (coefficient == CONDUCTIVITY) ? density[idx[0]] : 1.0/density[idx[0]];
      }
    });
  });//end of queue
}

// Initialises kx,ky
void cg_init_k(
        const int x, const int y, const int halo_depth, SyclBuffer& wBuff,
        SyclBuffer& kxBuff, SyclBuffer& kyBuff, const double rx, const double ry, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto kx           = kxBuff.get_access<access::mode::discard_write>(cgh);
    auto ky           = kyBuff.get_access<access::mode::discard_write>(cgh);
    auto w            = wBuff.get_access<access::mode::read>(cgh);

    //Define range
    auto myRange = range<1>(x*y);

    cgh.parallel_for<class cg_init_k>( myRange, [=] (id<1> idx){
      const size_t kk = idx[0] % x;
      const size_t jj = idx[0] / x;

      if(jj >= halo_depth && jj < y-1 &&
         kk >= halo_depth && kk < x-1)
      {
          kx[idx[0]] = rx*(w[idx[0]-1]+w[idx[0]]) /
              (2.0*w[idx[0]-1]*w[idx[0]]);
          ky[idx[0]] = ry*(w[idx[0]-x]+w[idx[0]]) /
              (2.0*w[idx[0]-x]*w[idx[0]]);
      }
    });
  });//end of queue
}

// Initialises w,r,p and calculates rro
void cg_init_others(
        const int x, const int y, const int halo_depth, SyclBuffer& kxBuff,
        SyclBuffer& kyBuff, SyclBuffer& pBuff, SyclBuffer& rBuff, SyclBuffer& uBuff, SyclBuffer& wBuff,
        double* rro, queue& device_queue)
{
  size_t wgroup_size = WORK_GROUP_SIZE; //define global
  auto len = rBuff.get_count();
  auto n_wgroups = floor((len + wgroup_size - 1) / wgroup_size);
  buffer<double, 1> tmpArrayBuff{range<1>{(size_t)n_wgroups}};

    device_queue.submit([&](handler &cgh) {
      auto r            = rBuff.get_access<access::mode::read_write>(cgh);
      auto w            = wBuff.get_access<access::mode::read_write>(cgh);
      auto u            = uBuff.get_access<access::mode::read>(cgh);
      auto p            = pBuff.get_access<access::mode::read_write>(cgh);
      auto kx           = kxBuff.get_access<access::mode::read>(cgh);
      auto ky           = kyBuff.get_access<access::mode::read>(cgh);
      auto global_mem = tmpArrayBuff.get_access<access::mode::discard_write>(cgh);
      accessor <double, 1, access::mode::read_write, access::target::local> local_mem(range<1>(wgroup_size), cgh);

      auto myRange = nd_range<1>(n_wgroups * wgroup_size, wgroup_size);
      cgh.parallel_for<class cg_init_others>( myRange, [=] (nd_item<1> item){

        size_t local_id = item.get_local_linear_id();
        size_t global_id = item.get_global_linear_id();
        local_mem[local_id] = 0;

        const size_t kk = global_id % x;
        const size_t jj = global_id / x;

        if(kk >= halo_depth && kk < x - halo_depth &&
           jj >= halo_depth && jj < y - halo_depth)
        {
            //smvp uses kx and ky and INDEX and dims.x and dims.y!!!!
            int index = global_id;
            const double smvp = SMVP(u);
            w[global_id] = smvp;
            r[global_id] = u[global_id]-w[global_id];
            p[global_id] = r[global_id];
            local_mem[local_id] = r[global_id]*p[global_id];
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

    *rro = SyclHelper::reduceArray(tmpArrayBuff, device_queue);
}

// Calculates the value for w
void cg_calc_w(
        const int x, const int y, const int halo_depth, SyclBuffer& wBuff,
        SyclBuffer& pBuff, SyclBuffer& kxBuff, SyclBuffer& kyBuff, double* pw, queue& device_queue)
{
  size_t wgroup_size = WORK_GROUP_SIZE; //define global
  auto len = wBuff.get_count();
  auto n_wgroups = floor((len + wgroup_size - 1) / wgroup_size);
  buffer<double, 1> tmpArrayBuff{range<1>{(size_t)n_wgroups}};

    device_queue.submit([&](handler &cgh) {
      auto w            = wBuff.get_access<access::mode::read_write>(cgh);
      auto p            = pBuff.get_access<access::mode::read>(cgh);
      auto kx           = kxBuff.get_access<access::mode::read>(cgh);
      auto ky           = kyBuff.get_access<access::mode::read>(cgh);
      auto global_mem = tmpArrayBuff.get_access<access::mode::discard_write>(cgh);
      accessor <double, 1, access::mode::read_write, access::target::local> local_mem(range<1>(wgroup_size), cgh);

      auto myRange = nd_range<1>(n_wgroups * wgroup_size, wgroup_size);

      cgh.parallel_for<class cg_calc_w>( myRange, [=] (nd_item<1> item){

        size_t local_id = item.get_local_linear_id();
        size_t global_id = item.get_global_linear_id();
        local_mem[local_id] = 0;

        const size_t kk = global_id % x;
        const size_t jj = global_id / x;

        if(kk >= halo_depth && kk < x - halo_depth &&
           jj >= halo_depth && jj < y - halo_depth)
        {
            //smvp uses kx and ky and INDEX and dims.x and dims.y!!!!
            int index = global_id;
            const double smvp = SMVP(p);
            w[global_id] = smvp;
            local_mem[local_id] =  w[global_id]*p[global_id];
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
    *pw = SyclHelper::reduceArray(tmpArrayBuff, device_queue);
}

// Calculates the value of u and r
void cg_calc_ur(
        const int x, const int y, const int halo_depth, SyclBuffer& uBuff,
        SyclBuffer& rBuff, SyclBuffer& pBuff, SyclBuffer& wBuff, const double alpha,
        double* rrn, queue& device_queue)
{
    size_t wgroup_size = WORK_GROUP_SIZE; //define global
    auto len = rBuff.get_count();
    auto n_wgroups = floor((len + wgroup_size - 1) / wgroup_size);
    buffer<double, 1> tmpArrayBuff{range<1>{(size_t)n_wgroups}};

    device_queue.submit([&](handler &cgh) {
      auto w           = wBuff.get_access<access::mode::read_write>(cgh);
      auto p           = pBuff.get_access<access::mode::read>(cgh);
      auto u           = uBuff.get_access<access::mode::read_write>(cgh);
      auto r           = rBuff.get_access<access::mode::read_write>(cgh);
      auto global_mem = tmpArrayBuff.get_access<access::mode::discard_write>(cgh);
      accessor <double, 1, access::mode::read_write, access::target::local> local_mem(range<1>(wgroup_size), cgh);

      auto myRange = nd_range<1>(n_wgroups * wgroup_size, wgroup_size);
      cgh.parallel_for<class cg_calc_ur>(myRange, [=] (nd_item<1> item){

        size_t local_id = item.get_local_linear_id();
        size_t global_id = item.get_global_linear_id();
        local_mem[local_id] = 0;

        const size_t kk = global_id % x;
        const size_t jj = global_id / x;

        if(kk >= halo_depth && kk < x - halo_depth &&
           jj >= halo_depth && jj < y - halo_depth)
        {
            u[global_id] += alpha*p[global_id];
            r[global_id] -= alpha*w[global_id];
            local_mem[local_id] = r[global_id]*r[global_id];
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

    *rrn = SyclHelper::reduceArray(tmpArrayBuff, device_queue);
}




// Calculates a value for p
void cg_calc_p(
        const int x, const int y, const int halo_depth, const double beta,
        SyclBuffer& pBuff, SyclBuffer& rBuff, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto p            = pBuff.get_access<access::mode::read_write>(cgh);
    auto r            = rBuff.get_access<access::mode::read>(cgh);

    //Define range
    auto myRange = range<1>(x*y);

    cgh.parallel_for<class cg_calc_p>( myRange, [=] (id<1> idx){
      const int kk = idx[0] % x;
      const int jj = idx[0] / x;

      if(kk >= halo_depth && kk < x - halo_depth &&
        jj >= halo_depth && jj < y - halo_depth) {
           p[idx[0]] = beta*p[idx[0]] + r[idx[0]];
     }
    });
  });//end of queue
}
