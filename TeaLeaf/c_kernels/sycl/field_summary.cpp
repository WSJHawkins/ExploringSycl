#include "sycl_shared.hpp"
#include "../../shared.h"

using namespace cl::sycl;

void field_summary_func(
	const int x, const int y, const int halo_depth, SyclBuffer& uBuff,
	SyclBuffer& densityBuff, SyclBuffer& energy0Buff, SyclBuffer& volumeBuff,
	double* vol, double* mass, double* ie, double* temp, queue& device_queue){

		double* tmpArrayVol  = new double[x*y];
		double* tmpArrayMass = new double[x*y];
		double* tmpArrayIe   = new double[x*y];
		double* tmpArrayTemp = new double[x*y];
		buffer<double, 1> tmpArrayBuffVol{tmpArrayVol, range<1>{(size_t)x*y}};
		buffer<double, 1> tmpArrayBuffMass{tmpArrayMass, range<1>{(size_t)x*y}};
		buffer<double, 1> tmpArrayBuffIe{tmpArrayIe, range<1>{(size_t)x*y}};
		buffer<double, 1> tmpArrayBuffTemp{tmpArrayTemp, range<1>{(size_t)x*y}};

	  device_queue.submit([&](handler &cgh) {
	    auto u                = uBuff.get_access<access::mode::read>(cgh);
	    auto density          = densityBuff.get_access<access::mode::read>(cgh);
	    auto energy0          = energy0Buff.get_access<access::mode::read>(cgh);
	    auto volume           = volumeBuff.get_access<access::mode::read>(cgh);
	    auto tmpArrayVolAcc   = tmpArrayBuffVol.get_access<access::mode::discard_write>(cgh);
			auto tmpArrayMassAcc  = tmpArrayBuffMass.get_access<access::mode::discard_write>(cgh);
			auto tmpArrayIeAcc    = tmpArrayBuffIe.get_access<access::mode::discard_write>(cgh);
			auto tmpArrayTempAcc  = tmpArrayBuffTemp.get_access<access::mode::discard_write>(cgh);

	    auto myRange = range<1>(x*y);

	    cgh.parallel_for<class field_summary_func>( myRange, [=] (id<1> idx){

	      const size_t kk = idx[0] % x;
	      const size_t jj = idx[0] / x;
	      if(kk >= halo_depth && kk < x - halo_depth &&
	         jj >= halo_depth && jj < y - halo_depth)
	      {
					const double cellVol = volume[idx[0]];
					const double cellMass = cellVol*density[idx[0]];
	        tmpArrayVolAcc[idx[0]]  = cellVol;
					tmpArrayMassAcc[idx[0]] = cellMass;
					tmpArrayIeAcc[idx[0]]   = cellMass*energy0[idx[0]];
					tmpArrayTempAcc[idx[0]] = cellMass*u[idx[0]];
	      }
	    });//end of parallel for
	  });//end of queue

	  *vol  = SyclHelper::reduceArray(tmpArrayBuffVol , device_queue);
		*mass = SyclHelper::reduceArray(tmpArrayBuffMass, device_queue);
		*ie   = SyclHelper::reduceArray(tmpArrayBuffIe  , device_queue);
		*temp = SyclHelper::reduceArray(tmpArrayBuffTemp, device_queue);
	}
