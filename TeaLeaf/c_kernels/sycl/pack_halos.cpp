#include "sycl_shared.hpp"
#include "../../shared.h"

using namespace cl::sycl;

// Packs the top halo buffer(s)
void pack_top(
        const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
        SyclBuffer& fieldBuff, const int depth, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto buffer          = bufferBuff.get_access<access::mode::write>(cgh);
    auto field           = fieldBuff.get_access<access::mode::read>(cgh);

    //Define range
    auto myRange = range<1>(x*depth);

    cgh.parallel_for<class pack_top>( myRange, [=] (id<1> idx){
      const int offset = x*(y-halo_depth-depth);
      buffer[idx[0]] = field[offset+idx[0]];
    });
  });//end of queue
}

// Packs the bottom halo buffer(s)
void pack_bottom(
        const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
        SyclBuffer& fieldBuff, const int depth, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto buffer          = bufferBuff.get_access<access::mode::write>(cgh);
    auto field           = fieldBuff.get_access<access::mode::read>(cgh);

    //Define range
    auto myRange = range<1>(x*depth);

    cgh.parallel_for<class pack_bottom>( myRange, [=] (id<1> idx){
      const int offset = x*halo_depth;
      buffer[idx[0]] = field[offset+idx[0]];
    });
  });//end of queue
}

// Packs the left halo buffer(s)
void pack_left(
        const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
        SyclBuffer& fieldBuff, const int depth, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto buffer          = bufferBuff.get_access<access::mode::write>(cgh);
    auto field           = fieldBuff.get_access<access::mode::read>(cgh);

    //Define range
    auto myRange = range<1>(y*depth);

    cgh.parallel_for<class pack_left>( myRange, [=] (id<1> idx){
      const int lines = idx[0]/depth;
      const int offset = halo_depth + lines*(x-depth);
      buffer[idx[0]] = field[offset+idx[0]];
    });
  });//end of queue
}

// Packs the right halo buffer(s)
void pack_right(
        const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
        SyclBuffer& fieldBuff, const int depth, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto buffer          = bufferBuff.get_access<access::mode::write>(cgh);
    auto field           = fieldBuff.get_access<access::mode::read>(cgh);

    //Define range
    auto myRange = range<1>(y*depth);

    cgh.parallel_for<class pack_right>( myRange, [=] (id<1> idx){
      const int lines = idx[0]/depth;
      const int offset = x-halo_depth-depth + lines*(x-depth);
      buffer[idx[0]] = field[offset+idx[0]];
    });
  });//end of queue
}

// Unpacks the top halo buffer(s)
void unpack_top(
        const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
        SyclBuffer& fieldBuff, const int depth, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto buffer          = bufferBuff.get_access<access::mode::read>(cgh);
    auto field           = fieldBuff.get_access<access::mode::write>(cgh);

    //Define range
    auto myRange = range<1>(x*depth);

    cgh.parallel_for<class unpack_top>( myRange, [=] (id<1> idx){
      const int offset = x*(y-halo_depth);
      field[offset+idx[0]]=buffer[idx[0]];
    });
  });//end of queue
}

// Unpacks the bottom halo buffer(s)
void unpack_bottom(
        const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
        SyclBuffer& fieldBuff, const int depth, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto buffer          = bufferBuff.get_access<access::mode::read>(cgh);
    auto field           = fieldBuff.get_access<access::mode::write>(cgh);

    //Define range
    auto myRange = range<1>(x*depth);

    cgh.parallel_for<class unpack_bottom>( myRange, [=] (id<1> idx){
      const int offset = x*(halo_depth-depth);
      field[offset+idx[0]]=buffer[idx[0]];
    });
  });//end of queue
}

// Unpacks the left halo buffer(s)
void unpack_left(
        const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
        SyclBuffer& fieldBuff, const int depth, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto buffer          = bufferBuff.get_access<access::mode::read>(cgh);
    auto field           = fieldBuff.get_access<access::mode::write>(cgh);

    //Define range
    auto myRange = range<1>(y*depth);

    cgh.parallel_for<class unpack_left>( myRange, [=] (id<1> idx){
      const int lines = idx[0]/depth;
      const int offset = halo_depth - depth + lines*(x-depth);
      field[offset+idx[0]]=buffer[idx[0]];
    });
  });//end of queue
}

// Unpacks the right halo buffer(s)
void unpack_right(
        const int x, const int y, const int halo_depth, SyclBuffer& bufferBuff,
        SyclBuffer& fieldBuff, const int depth, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto buffer          = bufferBuff.get_access<access::mode::read>(cgh);
    auto field           = fieldBuff.get_access<access::mode::write>(cgh);

    //Define range
    auto myRange = range<1>(y*depth);

    cgh.parallel_for<class unpack_right>( myRange, [=] (id<1> idx){
      const int lines = idx[0]/depth;
      const int offset = x-halo_depth + lines*(x-depth);
      field[offset+idx[0]]=buffer[idx[0]];
    });
  });//end of queue
}
