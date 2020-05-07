#include "sycl_shared.hpp"
#include "../../settings.h"
#include "../../shared.h"

using namespace cl::sycl;

// Initialises the vertices
void set_chunk_data_vertices(
			const int x, const int y, const int halo_depth, SyclBuffer& vertex_xBuff,
            SyclBuffer& vertex_yBuff, const double x_min, const double y_min,
            const double dx, const double dy, queue& device_queue)
{
	device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto vertex_x           = vertex_xBuff.get_access<access::mode::discard_write>(cgh);
		auto vertex_y           = vertex_yBuff.get_access<access::mode::discard_write>(cgh);

    //Define range
    auto myRange = range<1>(MAX(x,y)+1);

    cgh.parallel_for<class set_chunk_data_vertices>( myRange, [=] (id<1> idx){
			if(idx[0] < x+1)	{ vertex_x[idx[0]] = x_min+dx*(idx[0]-(double)halo_depth); }
			if(idx[0] < y+1)	{	vertex_y[idx[0]] = y_min+dy*(idx[0]-(double)halo_depth); }
    });
  });//end of queue
}

// Sets all of the cell data for a chunk
void set_chunk_data(
        const int x, const int y, const int halo_depth, SyclBuffer& vertex_xBuff,
        SyclBuffer& vertex_yBuff, SyclBuffer& cell_xBuff, SyclBuffer& cell_yBuff,
				SyclBuffer& volumeBuff, SyclBuffer& x_areaBuff, SyclBuffer& y_areaBuff,
				const double x_min, const double y_min, const double dx, const double dy,
				queue& device_queue)
{
	device_queue.submit([&](handler &cgh){
    //Set up accessors
    auto vertex_x           = vertex_xBuff.get_access<access::mode::read>(cgh);
		auto vertex_y           = vertex_yBuff.get_access<access::mode::read>(cgh);
		auto volume             = volumeBuff.get_access<access::mode::write>(cgh);
		auto x_area             = x_areaBuff.get_access<access::mode::write>(cgh);
		auto y_area             = y_areaBuff.get_access<access::mode::write>(cgh);
		auto cell_y             = cell_yBuff.get_access<access::mode::write>(cgh);
		auto cell_x             = cell_xBuff.get_access<access::mode::write>(cgh);

    //Define range
    auto myRange = range<1>(x*y);

    cgh.parallel_for<class set_chunk_data>( myRange, [=] (id<1> idx){
        if(idx[0] < x){ cell_x[idx[0]] = 0.5*(vertex_x[idx[0]]+vertex_x[idx[0]+1]); }

        if(idx[0] < y){ cell_y[idx[0]] = 0.5*(vertex_y[idx[0]]+vertex_y[idx[0]+1]); }

        if(idx[0] < x*y) {
         volume[idx[0]] = dx*dy;
         x_area[idx[0]] = dy;
         y_area[idx[0]] = dx;
        }
    });
  });//end of queue
}
