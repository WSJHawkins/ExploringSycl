#include <stdlib.h>
#include "sycl_shared.hpp"
#include "../../settings.h"
#include "../../shared.h"

using namespace cl::sycl;

// Allocates, and zeroes an individual buffer
void allocate_buffer(double** a, int x, int y)
{
    //*a = (double*)malloc(sizeof(double)*x*y);
    *a = new double[x*y];

    if(*a == NULL)
    {
        die(__LINE__, __FILE__, "Error allocating buffer %s\n");
    }

#pragma omp parallel for
    for(int jj = 0; jj < y; ++jj)
    {
        for(int kk = 0; kk < x; ++kk)
        {
            const int index = kk + jj*x;
            (*a)[index] = 0.0;
        }
    }
}

// Allocates all of the field buffers
void kernel_initialise(
        Settings* settings, int x, int y, SyclBuffer** density0,
        SyclBuffer** density, SyclBuffer** energy0, SyclBuffer** energy, SyclBuffer** u,
        SyclBuffer** u0, SyclBuffer** p, SyclBuffer** r, SyclBuffer** mi,
        SyclBuffer** w, SyclBuffer** kx, SyclBuffer** ky, SyclBuffer** sd,
        SyclBuffer** volume, SyclBuffer** x_area, SyclBuffer** y_area, SyclBuffer** cell_x,
        SyclBuffer** cell_y, SyclBuffer** cell_dx, SyclBuffer** cell_dy, SyclBuffer** vertex_dx,
        SyclBuffer** vertex_dy, SyclBuffer** vertex_x, SyclBuffer** vertex_y, SyclBuffer** comms_buffer,
        double** cg_alphas, double** cg_betas, double** cheby_alphas,
        double** cheby_betas, queue** device_queue)
{
    print_and_log(settings,
            "Performing this solve with the Sycl %s solver\n",
            settings->solver_name);

    (*device_queue) = new queue(cl::sycl::default_selector{});
    (*density0) = new SyclBuffer{range<1>{(size_t)x*y}};
    (*density) = new SyclBuffer{range<1>{(size_t)x*y}};
    (*energy0) = new SyclBuffer{range<1>{(size_t)x*y}};
    (*energy) = new SyclBuffer{range<1>{(size_t)x*y}};
    (*u) = new SyclBuffer{range<1>{(size_t)x*y}};
    (*u0) = new SyclBuffer{range<1>{(size_t)x*y}};
    (*p) = new SyclBuffer{range<1>{(size_t)x*y}};
    (*r) = new SyclBuffer{range<1>{(size_t)x*y}};
    (*mi) = new SyclBuffer{range<1>{(size_t)x*y}};
    (*w) = new SyclBuffer{range<1>{(size_t)x*y}};
    (*kx) = new SyclBuffer{range<1>{(size_t)x*y}};
    (*ky) = new SyclBuffer{range<1>{(size_t)x*y}};
    (*sd) = new SyclBuffer{range<1>{(size_t)x*y}};
    (*volume) = new SyclBuffer{range<1>{(size_t)x*y}};
    (*x_area) = new SyclBuffer{range<1>{(size_t)(x+1)*y}};
    (*y_area) = new SyclBuffer{range<1>{(size_t)x*(y+1)}};
    (*cell_x) = new SyclBuffer{range<1>{(size_t)x}};
    (*cell_y) = new SyclBuffer{range<1>{(size_t)y}};
    (*cell_dx) = new SyclBuffer{range<1>{(size_t)x}};
    (*cell_dy) = new SyclBuffer{range<1>{(size_t)y}};
    (*vertex_dx) = new SyclBuffer{range<1>{(size_t)(x+1)}};
    (*vertex_dy) = new SyclBuffer{range<1>{(size_t)(y+1)}};
    (*vertex_x) = new SyclBuffer{range<1>{(size_t)(x+1)}};
    (*vertex_y) = new SyclBuffer{range<1>{(size_t)(y+1)}};
    (*comms_buffer) = new SyclBuffer{range<1>{(size_t)(MAX(x, y)*settings->halo_depth)}};
    allocate_buffer(cg_alphas, settings->max_iters, 1);
    allocate_buffer(cg_betas, settings->max_iters, 1);
    allocate_buffer(cheby_alphas, settings->max_iters, 1);
    allocate_buffer(cheby_betas, settings->max_iters, 1);
}

void kernel_finalise(
  SyclBuffer** density0, SyclBuffer** density, SyclBuffer** energy0, SyclBuffer** energy, SyclBuffer** u,
  SyclBuffer** u0, SyclBuffer** p, SyclBuffer** r, SyclBuffer** mi,
  SyclBuffer** w, SyclBuffer** kx, SyclBuffer** ky, SyclBuffer** sd,
  SyclBuffer** volume, SyclBuffer** x_area, SyclBuffer** y_area, SyclBuffer** cell_x,
  SyclBuffer** cell_y, SyclBuffer** cell_dx, SyclBuffer** cell_dy, SyclBuffer** vertex_dx,
  SyclBuffer** vertex_dy, SyclBuffer** vertex_x, SyclBuffer** vertex_y, SyclBuffer** comms_buffer,
  double* cg_alphas, double* cg_betas, double* cheby_alphas,
  double* cheby_betas, cl::sycl::queue** device_queue)
{
    delete(cg_alphas);
    delete(cg_betas);
    delete(cheby_alphas);
    delete(cheby_betas);
    delete(*device_queue);
    delete(*density0);
    delete(*density);
    delete(*energy0);
    delete(*energy);
    delete(*u);
    delete(*u0);
    delete(*p);
    delete(*r);
    delete(*mi);
    delete(*w);
    delete(*kx);
    delete(*ky);
    delete(*sd);
    delete(*volume);
    delete(*x_area);
    delete(*y_area);
    delete(*cell_x);
    delete(*cell_y);
    delete(*cell_dx);
    delete(*cell_dy);
    delete(*vertex_dx);
    delete(*vertex_dy);
    delete(*vertex_x);
    delete(*vertex_y);
    delete(*comms_buffer);
}
