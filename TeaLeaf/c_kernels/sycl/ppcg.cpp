#include "sycl_shared.hpp"
#include "../../shared.h"

using namespace cl::sycl;

// Initialises Sd
void ppcg_init(
        const int x, const int y, const int halo_depth, const double theta,
        SyclBuffer& sdBuff, SyclBuffer& rBuff, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto sd           = sdBuff.get_access<access::mode::discard_write>(cgh);
    auto r            = rBuff.get_access<access::mode::read>(cgh);

    //Define range
    auto myRange = range<1>(x*y);

    cgh.parallel_for<class ppcg_init>( myRange, [=] (id<1> idx){
      const size_t kk = idx[0] % x;
      const size_t jj = idx[0] / x;
      if(kk >= halo_depth && kk < x - halo_depth &&
         jj >= halo_depth && jj < y - halo_depth)
      {
          sd[idx[0]] = r[idx[0]]/theta;
      }
    });
  });//end of queue
}

// Calculates U and R
void ppcg_calc_ur(
        const int x, const int y, const int halo_depth, SyclBuffer& sdBuff,
        SyclBuffer& rBuff, SyclBuffer& uBuff, SyclBuffer& kxBuff,
        SyclBuffer& kyBuff, queue& device_queue)
{
    device_queue.submit([&](handler &cgh){
      //Set up accessors
      auto sd           = sdBuff.get_access<access::mode::read>(cgh);
      auto r            = rBuff.get_access<access::mode::read_write>(cgh);
      auto u            = uBuff.get_access<access::mode::read_write>(cgh);
      auto kx           = kxBuff.get_access<access::mode::read>(cgh);
      auto ky           = kyBuff.get_access<access::mode::read>(cgh);

      //Define range
      auto myRange = range<1>(x*y);

      cgh.parallel_for<class ppcg_calc_ur>( myRange, [=] (id<1> idx){
        const size_t kk = idx[0] % x;
        const size_t jj = idx[0] / x;
        if(kk >= halo_depth && kk < x - halo_depth &&
           jj >= halo_depth && jj < y - halo_depth)
        {

            //smvp uses kx and ky and INDEX and dims.x and dims.y!!!!
            int index = idx[0];
            const double smvp = SMVP(sd);
            r[idx[0]] -= smvp;
            u[idx[0]] += sd[idx[0]];
        }
      });
    });//end of queue
}

// Calculates Sd
void ppcg_calc_sd(
        const int x, const int y, const int halo_depth, const double theta,
        const double alpha, const double beta, SyclBuffer& sdBuff,
        SyclBuffer& rBuff, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto sd           = sdBuff.get_access<access::mode::read_write>(cgh);
    auto r            = rBuff.get_access<access::mode::read>(cgh);

    //Define range
    auto myRange = range<1>(x*y);

    cgh.parallel_for<class ppcg_calc_sd>( myRange, [=] (id<1> idx){
      const size_t kk = idx[0] % x;
      const size_t jj = idx[0] / x;
      if(kk >= halo_depth && kk < x - halo_depth &&
         jj >= halo_depth && jj < y - halo_depth)
      {
          sd[idx[0]] = alpha*sd[idx[0]] + beta*r[idx[0]];
      }
    });
  });//end of queue
}
