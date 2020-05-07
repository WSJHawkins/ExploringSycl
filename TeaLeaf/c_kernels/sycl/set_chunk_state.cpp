#include "sycl_shared.hpp"
#include "../../shared.h"
#include "../../settings.h"

using namespace cl::sycl;

// Sets the initial state for the chunk
void set_chunk_initial_state(
            const int x, const int y, double default_energy,
            double default_density, SyclBuffer& energy0Buff,
            SyclBuffer& densityBuff, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto density      = densityBuff.get_access<access::mode::discard_write>(cgh);
    auto energy0       = energy0Buff.get_access<access::mode::discard_write>(cgh);

    //Define range
    auto myRange = range<1>(x*y);

    cgh.parallel_for<class set_chunk_initial_state>( myRange, [=] (id<1> idx){
      energy0[idx[0]] = default_energy;
      density[idx[0]] = default_density;
    });
  });//end of queue
}

// Sets all of the additional states in order
void set_chunk_state(
			const int x, const int y, const int halo_depth, State state,
            SyclBuffer& energy0Buff, SyclBuffer& densityBuff, SyclBuffer& uBuff,
            SyclBuffer& cell_xBuff, SyclBuffer& cell_yBuff,  SyclBuffer& vertex_xBuff,
            SyclBuffer& vertex_yBuff, queue& device_queue)
{
  device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto energy0        = energy0Buff.get_access<access::mode::read_write>(cgh);
    auto density        = densityBuff.get_access<access::mode::read_write>(cgh);
    auto u              = uBuff.get_access<access::mode::write>(cgh);
    auto cell_x         = cell_xBuff.get_access<access::mode::read>(cgh);
    auto cell_y         = cell_yBuff.get_access<access::mode::read>(cgh);
    auto vertex_x       = vertex_xBuff.get_access<access::mode::read>(cgh);
    auto vertex_y       = vertex_yBuff.get_access<access::mode::read>(cgh);

    //Define range
    auto myRange = range<1>(x*y);

    cgh.parallel_for<class set_chunk_state>( myRange, [=] (id<1> idx){
      const size_t kk = idx[0] % x;
      const size_t jj = idx[0] / x;
      bool applyState = false;

      if(state.geometry == RECTANGULAR) // Rectangular state
      {
          applyState = (
                  vertex_x[kk+1] >= state.x_min &&
                  vertex_x[kk] < state.x_max  &&
                  vertex_y[jj+1] >= state.y_min &&
                  vertex_y[jj] < state.y_max);
      }
      else if(state.geometry == CIRCULAR) // Circular state
      {
          double radius = cl::sycl::sqrt(
                  (cell_x[kk]-state.x_min)*(cell_x[kk]-state.x_min)+
                  (cell_y[jj]-state.y_min)*(cell_y[jj]-state.y_min));

          applyState = (radius <= state.radius);
      }
      else if(state.geometry == POINT) // Point state
      {
          applyState = (
                  vertex_x[kk] == state.x_min &&
                  vertex_y[jj] == state.y_min);
      }

      // Check if state applies at this vertex, and apply
      if(applyState)
      {
          energy0[idx[0]] = state.energy;
          density[idx[0]] = state.density;
      }

      if(kk > 0 && kk < x-1 && jj > 0 && jj < y-1)
      {
          u[idx[0]] = energy0[idx[0]]*density[idx[0]];
      }
    });
  });//end of queue
}
