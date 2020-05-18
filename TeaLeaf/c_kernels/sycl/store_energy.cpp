#include "sycl_shared.hpp"

using namespace cl::sycl;

// Copies energy0 into energy1.
void store_energy(
  const int x, const int y, SyclBuffer& energyBuff, SyclBuffer& energy0Buff,
  queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    auto energy       = energyBuff.get_access<access::mode::write>(cgh);
    auto energy0       = energy0Buff.get_access<access::mode::read>(cgh);

    auto myRange = range<1>(x*y);
    cgh.parallel_for<class store_energy>( myRange, [=] (id<1> idx){

		    energy[idx[0]] = energy0[idx[0]];

    });//end of parallel for
  });//end of queue
  #ifdef ENABLE_PROFILING
  device_queue.wait();
  #endif
}
