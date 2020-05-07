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
  double* tmpArray = new double[x*y];
	buffer<double, 1> tmpArrayBuff{tmpArray, range<1>{(size_t)x*y}};

  device_queue.submit([&](handler &cgh) {
    auto buffer           = bufferBuff.get_access<access::mode::read>(cgh);
    auto tmpArray     = tmpArrayBuff.get_access<access::mode::discard_write>(cgh);

    auto myRange = range<1>(x*y);

    cgh.parallel_for<class calculate_2norm>( myRange, [=] (id<1> idx){
      tmpArray[idx[0]] = 0;
      const size_t kk = idx[0] % x;
      const size_t jj = idx[0] / x;
      if(kk >= halo_depth && kk < x - halo_depth &&
         jj >= halo_depth && jj < y - halo_depth)
      {
          tmpArray[idx[0]] = buffer[idx[0]]*buffer[idx[0]];
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
