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

	size_t wgroup_size = WORK_GROUP_SIZE; //define global
	auto len = uBuff.get_count();
	auto n_wgroups = floor((len + wgroup_size - 1) / wgroup_size);
	buffer<double, 1> tmpArrayBuff{range<1>{(size_t)n_wgroups}};

  device_queue.submit([&](handler &cgh) {
    auto r            = rBuff.get_access<access::mode::read>(cgh);
    auto u            = uBuff.get_access<access::mode::read_write>(cgh);
    auto u0           = u0Buff.get_access<access::mode::read>(cgh);
    auto kx           = kxBuff.get_access<access::mode::read>(cgh);
    auto ky           = kyBuff.get_access<access::mode::read>(cgh);
		auto global_mem = tmpArrayBuff.get_access<access::mode::discard_write>(cgh);
		accessor <double, 1, access::mode::read_write, access::target::local> local_mem(range<1>(wgroup_size), cgh);

    auto myRange = nd_range<1>(n_wgroups * wgroup_size, wgroup_size);
    cgh.parallel_for<class jacobi_iterate>(myRange, [=] (nd_item<1> item){

			size_t local_id = item.get_local_linear_id();
			size_t global_id = item.get_global_linear_id();
			local_mem[local_id] = 0;

			const size_t kk = global_id % x;
			const size_t jj = global_id / x;
      if(kk >= halo_depth && kk < x - halo_depth &&
         jj >= halo_depth && jj < y - halo_depth)
      {
        u[global_id] = (u0[global_id]
            + (kx[global_id+1]*r[global_id+1] + kx[global_id]*r[global_id-1])
            + (ky[global_id+x]*r[global_id+x] + ky[global_id]*r[global_id-x]))
            / (1.0 + (kx[global_id]+kx[global_id+1]) + (ky[global_id]+ky[global_id+x]));

        local_mem[local_id] += cl::sycl::fabs((u[global_id]-r[global_id])); // fabs is float version of abs
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

  *error = SyclHelper::reduceArray(tmpArrayBuff, device_queue);
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
