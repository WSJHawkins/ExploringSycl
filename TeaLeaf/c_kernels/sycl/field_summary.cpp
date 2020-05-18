#include "sycl_shared.hpp"
#include "../../shared.h"

using namespace cl::sycl;

void field_summary_func(
	const int x, const int y, const int halo_depth, SyclBuffer& uBuff,
	SyclBuffer& densityBuff, SyclBuffer& energy0Buff, SyclBuffer& volumeBuff,
	double* vol, double* mass, double* ie, double* temp, queue& device_queue)
{

	size_t wgroup_size = WORK_GROUP_SIZE;
  auto len = uBuff.get_count();
  auto n_wgroups = floor((len + wgroup_size - 1) / wgroup_size);
	buffer<double, 1> tmpArrayBuffVol{range<1>{(size_t)n_wgroups}};
	buffer<double, 1> tmpArrayBuffMass{range<1>{(size_t)n_wgroups}};
	buffer<double, 1> tmpArrayBuffIe{range<1>{(size_t)n_wgroups}};
	buffer<double, 1> tmpArrayBuffTemp{range<1>{(size_t)n_wgroups}};

	device_queue.submit([&](handler &cgh) {
		auto u                = uBuff.get_access<access::mode::read>(cgh);
	  auto density          = densityBuff.get_access<access::mode::read>(cgh);
	  auto energy0          = energy0Buff.get_access<access::mode::read>(cgh);
	  auto volume           = volumeBuff.get_access<access::mode::read>(cgh);
	  auto tmpArrayVolAcc   = tmpArrayBuffVol.get_access<access::mode::discard_write>(cgh);
		auto tmpArrayMassAcc  = tmpArrayBuffMass.get_access<access::mode::discard_write>(cgh);
		auto tmpArrayIeAcc    = tmpArrayBuffIe.get_access<access::mode::discard_write>(cgh);
		auto tmpArrayTempAcc  = tmpArrayBuffTemp.get_access<access::mode::discard_write>(cgh);
		accessor <double, 1, access::mode::read_write, access::target::local> local_memVol(range<1>(wgroup_size), cgh);
		accessor <double, 1, access::mode::read_write, access::target::local> local_memMass(range<1>(wgroup_size), cgh);
		accessor <double, 1, access::mode::read_write, access::target::local> local_memIe(range<1>(wgroup_size), cgh);
		accessor <double, 1, access::mode::read_write, access::target::local> local_memTemp(range<1>(wgroup_size), cgh);

	  auto myRange = nd_range<1>(n_wgroups * wgroup_size, wgroup_size);
	  cgh.parallel_for<class field_summary_func>( myRange, [=] (nd_item<1> item){

			size_t local_id = item.get_local_linear_id();
      size_t global_id = item.get_global_linear_id();
      local_memVol[local_id] = 0;
			local_memMass[local_id] = 0;
			local_memIe[local_id] = 0;
			local_memTemp[local_id] = 0;

      const size_t kk = global_id % x;
      const size_t jj = global_id / x;
	    if(kk >= halo_depth && kk < x - halo_depth &&
	    	jj >= halo_depth && jj < y - halo_depth)
	    {
				const double cellVol = volume[global_id];
				const double cellMass = cellVol*density[global_id];
	      local_memVol[local_id]  = cellVol;
				local_memMass[local_id] = cellMass;
				local_memIe[local_id]   = cellMass*energy0[global_id];
				local_memTemp[local_id] = cellMass*u[global_id];
	     }

			item.barrier(access::fence_space::local_space);

      for (size_t stride = 1; stride < wgroup_size; stride *= 2) {
      	auto idx = 2 * stride * local_id;
        if (idx < wgroup_size) {
        	local_memVol[idx] = local_memVol[idx] + local_memVol[idx + stride];
					local_memMass[idx] = local_memMass[idx] + local_memMass[idx + stride];
					local_memIe[idx] = local_memIe[idx] + local_memIe[idx + stride];
					local_memTemp[idx] = local_memTemp[idx] + local_memTemp[idx + stride];
        }
				item.barrier(access::fence_space::local_space);
      }

      if (local_id == 0) {
      	tmpArrayVolAcc[item.get_group_linear_id()] = local_memVol[0];
				tmpArrayMassAcc[item.get_group_linear_id()] = local_memMass[0];
				tmpArrayIeAcc[item.get_group_linear_id()] = local_memIe[0];
				tmpArrayTempAcc[item.get_group_linear_id()] = local_memTemp[0];
      }

	    });//end of parallel for
	  });//end of queue
		#ifdef ENABLE_PROFILING
		device_queue.wait();
		#endif

	 len = tmpArrayBuffVol.get_count();
	while (len != 1) {
		n_wgroups = floor((len + wgroup_size - 1) / wgroup_size);

		device_queue.submit([&] (handler& cgh) {
			auto tmpArrayVolAcc   = tmpArrayBuffVol.get_access<access::mode::read_write>(cgh);
			auto tmpArrayMassAcc  = tmpArrayBuffMass.get_access<access::mode::read_write>(cgh);
			auto tmpArrayIeAcc    = tmpArrayBuffIe.get_access<access::mode::read_write>(cgh);
			auto tmpArrayTempAcc  = tmpArrayBuffTemp.get_access<access::mode::read_write>(cgh);
			accessor <double, 1, access::mode::read_write, access::target::local> local_memVol(range<1>(wgroup_size), cgh);
			accessor <double, 1, access::mode::read_write, access::target::local> local_memMass(range<1>(wgroup_size), cgh);
			accessor <double, 1, access::mode::read_write, access::target::local> local_memIe(range<1>(wgroup_size), cgh);
			accessor <double, 1, access::mode::read_write, access::target::local> local_memTemp(range<1>(wgroup_size), cgh);

			auto myRange = nd_range<1>(n_wgroups * wgroup_size, wgroup_size);
			cgh.parallel_for<class field_summary_func_reduction>(myRange,[=] (nd_item<1> item) {

				size_t local_id = item.get_local_linear_id();
				size_t global_id = item.get_global_linear_id();
				local_memVol[local_id] = 0;
				local_memMass[local_id] = 0;
				local_memIe[local_id] = 0;
				local_memTemp[local_id] = 0;

				if(global_id+1<=len){
					local_memVol[local_id]  = tmpArrayVolAcc[global_id];
					local_memMass[local_id] = tmpArrayMassAcc[global_id];
					local_memIe[local_id]   = tmpArrayIeAcc[global_id];
					local_memTemp[local_id] = tmpArrayTempAcc[global_id];
				}

				item.barrier(access::fence_space::local_space);

				for (size_t stride = 1; stride < wgroup_size; stride *= 2) {
			  	auto idx = 2 * stride * local_id;
			    if (idx < wgroup_size) {
			    	local_memVol[idx] = local_memVol[idx] + local_memVol[idx + stride];
						local_memMass[idx] = local_memMass[idx] + local_memMass[idx + stride];
						local_memIe[idx] = local_memIe[idx] + local_memIe[idx + stride];
						local_memTemp[idx] = local_memTemp[idx] + local_memTemp[idx + stride];
					}
					item.barrier(access::fence_space::local_space);
				}

				if (local_id == 0) {
			  	tmpArrayVolAcc[item.get_group_linear_id()] = local_memVol[0];
					tmpArrayMassAcc[item.get_group_linear_id()] = local_memMass[0];
					tmpArrayIeAcc[item.get_group_linear_id()] = local_memIe[0];
					tmpArrayTempAcc[item.get_group_linear_id()] = local_memTemp[0];
				}

			});//end of parallel for
	  });//end of queue
		#ifdef ENABLE_PROFILING
		device_queue.wait();
		#endif

		len = n_wgroups;
	}//end of while

	auto accVol  = tmpArrayBuffVol.get_access<access::mode::read>();
	auto accMass = tmpArrayBuffMass.get_access<access::mode::read>();
	auto accIe   = tmpArrayBuffIe.get_access<access::mode::read>();
	auto accTemp = tmpArrayBuffTemp.get_access<access::mode::read>();
	*vol  = accVol[0];
	*mass = accMass[0];
	*ie   = accIe[0];
	*temp = accTemp[0];
}
