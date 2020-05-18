#include "sycl_shared.hpp"
#include "../../shared.h"

using namespace cl::sycl;

// Updates the local left halo region(s)
void update_left(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  const int face, const int depth, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    auto buffer = bufferBuff.get_access<access::mode::read_write>(cgh);

    auto myRange = range<1>(y*depth);
    cgh.parallel_for<class update_left>( myRange, [=] (id<1> idx){

      const size_t flip = idx[0] % depth;
      const size_t lines = idx[0]/depth;
      const size_t offset = lines*(x - depth);
      const size_t to_index = offset + halo_depth - depth + idx[0];
      const size_t from_index = to_index + 2*(depth - flip) - 1;
      buffer[to_index] = buffer[from_index];

    });//end of parallel for
  });//end of queue
  #ifdef ENABLE_PROFILING
  device_queue.wait();
  #endif
}

// Updates the local right halo region(s)
void update_right(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  const int face, const int depth, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    auto buffer = bufferBuff.get_access<access::mode::read_write>(cgh);

    auto myRange = range<1>(y*depth);
    cgh.parallel_for<class update_right>( myRange, [=] (id<1> idx){

      const size_t flip = idx[0] % depth;
      const size_t lines = idx[0]/depth;
      const size_t offset = x-halo_depth + lines*(x-depth);
      const size_t to_index = offset+idx[0];
      const size_t from_index = to_index-(1+flip*2);
      buffer[to_index] = buffer[from_index];

    });//end of parallel for
  });//end of queue
  #ifdef ENABLE_PROFILING
  device_queue.wait();
  #endif
}

// Updates the local top halo region(s)
void update_top(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  const int face, const int depth, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
  auto buffer = bufferBuff.get_access<access::mode::read_write>(cgh);

  auto myRange = range<1>(x*depth);
  cgh.parallel_for<class update_top>( myRange, [=] (id<1> idx){

    const size_t lines = idx[0]/x;
    const size_t offset = x*(y-halo_depth);
    const size_t to_index = offset+idx[0];
    const size_t from_index = to_index-(1+lines*2)*x;
    buffer[to_index] = buffer[from_index];

    });//end of parallel for
  });//end of queue
  #ifdef ENABLE_PROFILING
  device_queue.wait();
  #endif
}

// Updates the local bottom halo region(s)
void update_bottom(
  const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
  const int face, const int depth, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    auto buffer = bufferBuff.get_access<access::mode::read_write>(cgh);

    auto myRange = range<1>(x*depth);
    cgh.parallel_for<class update_bottom>( myRange, [=] (id<1> idx){

      const size_t lines = idx[0]/x;
      const size_t offset = x*halo_depth;
      const size_t from_index = offset+idx[0];
      const size_t to_index = from_index-(1+lines*2)*x;
      buffer[to_index] = buffer[from_index];

    });//end of parallel for
  });//end of queue
  #ifdef ENABLE_PROFILING
  device_queue.wait();
  #endif
}
