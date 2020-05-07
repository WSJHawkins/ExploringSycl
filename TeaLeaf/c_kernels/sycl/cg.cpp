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
    double* tmpArray = new double[x*y];
    buffer<double, 1> tmpArrayBuff{tmpArray, range<1>{(size_t)x*y}};
    device_queue.submit([&](handler &cgh) {
      auto r            = rBuff.get_access<access::mode::read_write>(cgh);
      auto w            = wBuff.get_access<access::mode::read_write>(cgh);
      auto u            = uBuff.get_access<access::mode::read>(cgh);
      auto p            = pBuff.get_access<access::mode::read_write>(cgh);
      auto kx           = kxBuff.get_access<access::mode::read>(cgh);
      auto ky           = kyBuff.get_access<access::mode::read>(cgh);
      auto tmpArrayAcc     = tmpArrayBuff.get_access<access::mode::discard_write>(cgh);

      auto myRange = range<1>(x*y);

      cgh.parallel_for<class cg_init_others>( myRange, [=] (id<1> idx){

        tmpArrayAcc[idx[0]] = 0;

        const size_t kk = idx[0] % x;
        const size_t jj = idx[0] / x;

        if(kk >= halo_depth && kk < x - halo_depth &&
           jj >= halo_depth && jj < y - halo_depth)
        {
            //smvp uses kx and ky and INDEX and dims.x and dims.y!!!!
            int index = idx[0];
            const double smvp = SMVP(u);
            w[idx[0]] = smvp;
            r[idx[0]] = u[idx[0]]-w[idx[0]];
            p[idx[0]] = r[idx[0]];
            tmpArrayAcc[idx[0]] = r[idx[0]]*p[idx[0]];
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
    double* tmpArray = new double[x*y];
    buffer<double, 1> tmpArrayBuff{tmpArray, range<1>{(size_t)x*y}};
    device_queue.submit([&](handler &cgh) {
      auto w            = wBuff.get_access<access::mode::read_write>(cgh);
      auto p            = pBuff.get_access<access::mode::read>(cgh);
      auto kx           = kxBuff.get_access<access::mode::read>(cgh);
      auto ky           = kyBuff.get_access<access::mode::read>(cgh);
      auto tmpArrayAcc     = tmpArrayBuff.get_access<access::mode::discard_write>(cgh);

      auto myRange = range<1>(x*y);

      cgh.parallel_for<class cg_calc_w>( myRange, [=] (id<1> idx){

        tmpArrayAcc[idx[0]] = 0;

        const size_t kk = idx[0] % x;
        const size_t jj = idx[0] / x;

        if(kk >= halo_depth && kk < x - halo_depth &&
           jj >= halo_depth && jj < y - halo_depth)
        {
            //smvp uses kx and ky and INDEX and dims.x and dims.y!!!!
            int index = idx[0];
            const double smvp = SMVP(p);
            w[idx[0]] = smvp;
            tmpArrayAcc[idx[0]] =  w[idx[0]]*p[idx[0]];
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
    double* tmpArray = new double[x*y];
    buffer<double, 1> tmpArrayBuff{tmpArray, range<1>{(size_t)x*y}};
    device_queue.submit([&](handler &cgh) {
      auto w           = wBuff.get_access<access::mode::read_write>(cgh);
      auto p           = pBuff.get_access<access::mode::read>(cgh);
      auto u           = uBuff.get_access<access::mode::read_write>(cgh);
      auto r           = rBuff.get_access<access::mode::read_write>(cgh);
      auto tmpArray    = tmpArrayBuff.get_access<access::mode::discard_write>(cgh);

      auto myRange = range<1>(x*y);

      cgh.parallel_for<class cg_calc_ur>( myRange, [=] (id<1> idx){

        tmpArray[idx[0]] = 0;

        const size_t kk = idx[0] % x;
        const size_t jj = idx[0] / x;

        if(kk >= halo_depth && kk < x - halo_depth &&
           jj >= halo_depth && jj < y - halo_depth)
        {
            u[idx[0]] += alpha*p[idx[0]];
            r[idx[0]] -= alpha*w[idx[0]];
            tmpArray[idx[0]] = r[idx[0]]*r[idx[0]];
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
