#pragma once

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

#define WORK_GROUP_SIZE 64
typedef buffer<double, 1> SyclBuffer;

class SyclHelper
{
	public:
    // Used to manage copying the raw pointers
		template <class Type>
		static void PackMirror(SyclBuffer& buffToPack, const Type* buffer, int len)
		{
			auto hostAcc = buffToPack.get_access<access::mode::write>();
			for(int kk = 0; kk != len; ++kk)
			{
				hostAcc[kk] = buffer[kk];
			}
		}

		template <class Type>
		static void UnpackMirror(Type* buffer, SyclBuffer& buffToUnpack, int len)
		{
			auto hostAcc = buffToUnpack.get_access<access::mode::read>();
			for(int kk = 0; kk != len; ++kk)
			{
				buffer[kk] = hostAcc[kk];
			}
		}

		static double reduceArray(SyclBuffer& arrayBuff, queue& device_queue)
		{
		  size_t wgroup_size = WORK_GROUP_SIZE; //define global
		  auto len = arrayBuff.get_count();
		  while (len != 1) {
		       auto n_wgroups = floor((len + wgroup_size - 1) / wgroup_size);
		       device_queue.submit([&] (handler& cgh) {
		          accessor <double, 1, access::mode::read_write, access::target::local>
		                         local_mem(range<1>(wgroup_size), cgh);

		          auto global_mem = arrayBuff.get_access<access::mode::read_write>(cgh);
		          cgh.parallel_for<class reduceArrays>(
		               nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
		               [=] (nd_item<1> item) {

		            size_t local_id = item.get_local_linear_id();
		            size_t global_id = item.get_global_linear_id();
		            local_mem[local_id] = 0;

		            if(global_id+1<=len){
		                local_mem[local_id] = global_mem[global_id];
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
		          });
		       });
		    len = n_wgroups;
		  }
		  auto acc = arrayBuff.get_access<access::mode::read>();
		  return acc[0];
		}

		static void GetSyclArrayVal(
						int len, SyclBuffer& device_buffer, const char* name)
		{
				auto hostAcc = device_buffer.get_access<access::mode::read>();

				double temp = 0.0;
				for(int ii = 0; ii < len; ++ii)
				{
						temp += hostAcc[ii];
				}

				printf("Array %s is:", name);
				for (int n = 0; n < len; n++){
					printf(" %.3e ,", hostAcc[n]);
				}
				printf("\n");
		}
};
