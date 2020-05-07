#include "sycl_shared.hpp"
#include "../../shared.h"

using namespace cl::sycl;

// Initialises the Jacobi solver
void jacobi_init(
			const int x, const int y, const int halo_depth,
            const int coefficient, const double rx, const double ry, SyclBuffer& uBuff,
            SyclBuffer& u0Buff, SyclBuffer& densityBuff, SyclBuffer& energyBuff,
						 SyclBuffer& kxBuff, SyclBuffer& kyBuff, queue& device_queue)
{
	device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto u            = uBuff.get_access<access::mode::write>(cgh);
		auto u0           = u0Buff.get_access<access::mode::read_write>(cgh);
    auto density      = densityBuff.get_access<access::mode::read>(cgh);
    auto energy       = energyBuff.get_access<access::mode::read>(cgh);
		auto kx           = kxBuff.get_access<access::mode::write>(cgh);
		auto ky           = kyBuff.get_access<access::mode::write>(cgh);

    //Define range
    auto myRange = range<1>(x*y);

    cgh.parallel_for<class jacobi_init>( myRange, [=] (id<1> idx){
      const size_t kk = idx[0] % x;
      const size_t jj = idx[0] / x;

			if(kk > 0 && kk < x - 1 &&
				 jj > 0 && jj < y - 1)
			{
					u0[idx[0]] = energy[idx[0]]*density[idx[0]];
					u[idx[0]] = u0[idx[0]];
			}

			if(jj >= halo_depth && jj < y-1 &&
				 kk >= halo_depth && kk < x-1)
			{
					double densityCentre = (coefficient == CONDUCTIVITY)
							? density[idx[0]] : 1.0/density[idx[0]];
					double densityLeft = (coefficient == CONDUCTIVITY)
							? density[idx[0]-1] : 1.0/density[idx[0]-1];
					double densityDown = (coefficient == CONDUCTIVITY)
							? density[idx[0]-x] : 1.0/density[idx[0]-x];

					kx[idx[0]] = rx*(densityLeft+densityCentre) /
							(2.0*densityLeft*densityCentre);
					ky[idx[0]] = ry*(densityDown+densityCentre) /
							(2.0*densityDown*densityCentre);
			}
    });
  });//end of queue
}

// Main Jacobi solver method.
void jacobi_iterate(
        const int x, const int y, const int halo_depth, SyclBuffer& uBuff,
        SyclBuffer& u0Buff, SyclBuffer& rBuff, SyclBuffer& kxBuff, SyclBuffer& kyBuff,
        double* error, queue& device_queue)
{

  double* tmpArray = new double[x*y];
	buffer<double, 1> tmpArrayBuff{tmpArray, range<1>{(size_t)x*y}};

  device_queue.submit([&](handler &cgh) {
    auto r            = rBuff.get_access<access::mode::read>(cgh);
    auto u            = uBuff.get_access<access::mode::read_write>(cgh);
    auto u0           = u0Buff.get_access<access::mode::read>(cgh);
    auto kx           = kxBuff.get_access<access::mode::read>(cgh);
    auto ky           = kyBuff.get_access<access::mode::read>(cgh);
    auto tmpArrayAcc     = tmpArrayBuff.get_access<access::mode::discard_write>(cgh);

    auto myRange = range<1>(x*y);

    cgh.parallel_for<class jacobi_iterate>( myRange, [=] (id<1> idx){

      tmpArrayAcc[idx[0]] = 0;

      const size_t kk = idx[0] % x;
      const size_t jj = idx[0] / x;
      if(kk >= halo_depth && kk < x - halo_depth &&
         jj >= halo_depth && jj < y - halo_depth)
      {
        u[idx[0]] = (u0[idx[0]]
            + (kx[idx[0]+1]*r[idx[0]+1] + kx[idx[0]]*r[idx[0]-1])
            + (ky[idx[0]+x]*r[idx[0]+x] + ky[idx[0]]*r[idx[0]-x]))
            / (1.0 + (kx[idx[0]]+kx[idx[0]+1]) + (ky[idx[0]]+ky[idx[0]+x]));

        tmpArrayAcc[idx[0]] += cl::sycl::fabs((u[idx[0]]-r[idx[0]])); // fabs is float version of abs
      }
    });//end of parallel for
  });//end of queue

  *error += SyclHelper::reduceArray(tmpArrayBuff, device_queue);
}

// Copies u into r
void jacobi_copy_u(
	const int x, const int y, SyclBuffer& rBuff, SyclBuffer& uBuff, queue& device_queue)
{
	device_queue.submit([&](handler &cgh){
		//Set up accessors
		auto r            = rBuff.get_access<access::mode::write>(cgh);
		auto u            = uBuff.get_access<access::mode::read>(cgh);

		//Define range
		auto myRange = range<1>(x*y);

		cgh.parallel_for<class jacobi_copy_u>( myRange, [=] (id<1> idx){
			r[idx[0]] = u[idx[0]];
		});
	});//end of queue
}
