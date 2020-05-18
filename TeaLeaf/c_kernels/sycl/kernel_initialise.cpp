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
  Settings* settings, int x, int y, SyclBuffer** density0Buff,
  SyclBuffer** densityBuff, SyclBuffer** energy0Buff, SyclBuffer** energyBuff,
  SyclBuffer** uBuff, SyclBuffer** u0Buff, SyclBuffer** pBuff, SyclBuffer** rBuff,
  SyclBuffer** miBuff, SyclBuffer** wBuff, SyclBuffer** kxBuff, SyclBuffer** kyBuff,
  SyclBuffer** sdBuff, SyclBuffer** volumeBuff, SyclBuffer** x_areaBuff,
  SyclBuffer** y_areaBuff, SyclBuffer** cell_xBuff, SyclBuffer** cell_yBuff,
  SyclBuffer** cell_dxBuff, SyclBuffer** cell_dyBuff, SyclBuffer** vertex_dxBuff,
  SyclBuffer** vertex_dyBuff, SyclBuffer** vertex_xBuff, SyclBuffer** vertex_yBuff,
  SyclBuffer** comms_bufferBuff, double** cg_alphas, double** cg_betas,
  double** cheby_alphas, double** cheby_betas, queue** device_queue)
{
    print_and_log(settings,
      "Performing this solve with the Sycl %s solver\n",
      settings->solver_name);

    (*device_queue) = new queue(cl::sycl::default_selector{});
    std::cout << "Running on " << (**device_queue).get_device().get_info<cl::sycl::info::device::name>()  << "\n";

    (*density0Buff)     = new SyclBuffer{range<1>{(size_t)x*y}};
    (*densityBuff)      = new SyclBuffer{range<1>{(size_t)x*y}};
    (*energy0Buff)      = new SyclBuffer{range<1>{(size_t)x*y}};
    (*energyBuff)       = new SyclBuffer{range<1>{(size_t)x*y}};
    (*uBuff)            = new SyclBuffer{range<1>{(size_t)x*y}};
    (*u0Buff)           = new SyclBuffer{range<1>{(size_t)x*y}};
    (*pBuff)            = new SyclBuffer{range<1>{(size_t)x*y}};
    (*rBuff)            = new SyclBuffer{range<1>{(size_t)x*y}};
    (*miBuff)           = new SyclBuffer{range<1>{(size_t)x*y}};
    (*wBuff)            = new SyclBuffer{range<1>{(size_t)x*y}};
    (*kxBuff)           = new SyclBuffer{range<1>{(size_t)x*y}};
    (*kyBuff)           = new SyclBuffer{range<1>{(size_t)x*y}};
    (*sdBuff)           = new SyclBuffer{range<1>{(size_t)x*y}};
    (*volumeBuff)       = new SyclBuffer{range<1>{(size_t)x*y}};
    (*x_areaBuff)       = new SyclBuffer{range<1>{(size_t)(x+1)*y}};
    (*y_areaBuff)       = new SyclBuffer{range<1>{(size_t)x*(y+1)}};
    (*cell_xBuff)       = new SyclBuffer{range<1>{(size_t)x}};
    (*cell_yBuff)       = new SyclBuffer{range<1>{(size_t)y}};
    (*cell_dxBuff)      = new SyclBuffer{range<1>{(size_t)x}};
    (*cell_dyBuff)      = new SyclBuffer{range<1>{(size_t)y}};
    (*vertex_dxBuff)    = new SyclBuffer{range<1>{(size_t)(x+1)}};
    (*vertex_dyBuff)    = new SyclBuffer{range<1>{(size_t)(y+1)}};
    (*vertex_xBuff)     = new SyclBuffer{range<1>{(size_t)(x+1)}};
    (*vertex_yBuff)     = new SyclBuffer{range<1>{(size_t)(y+1)}};
    (*comms_bufferBuff) = new SyclBuffer{range<1>{(size_t)(MAX(x, y)*settings->halo_depth)}};

    allocate_buffer(cg_alphas, settings->max_iters, 1);
    allocate_buffer(cg_betas, settings->max_iters, 1);
    allocate_buffer(cheby_alphas, settings->max_iters, 1);
    allocate_buffer(cheby_betas, settings->max_iters, 1);
}

void kernel_finalise(
  SyclBuffer** density0Buff, SyclBuffer** densityBuff, SyclBuffer** energy0Buff,
  SyclBuffer** energyBuff, SyclBuffer** uBuff, SyclBuffer** u0Buff, SyclBuffer** pBuff,
  SyclBuffer** rBuff, SyclBuffer** miBuff, SyclBuffer** wBuff, SyclBuffer** kxBuff,
  SyclBuffer** kyBuff, SyclBuffer** sdBuff, SyclBuffer** volumeBuff, SyclBuffer** x_areaBuff,
  SyclBuffer** y_areaBuff, SyclBuffer** cell_xBuff, SyclBuffer** cell_yBuff,
  SyclBuffer** cell_dxBuff, SyclBuffer** cell_dyBuff, SyclBuffer** vertex_dxBuff,
  SyclBuffer** vertex_dyBuff, SyclBuffer** vertex_xBuff, SyclBuffer** vertex_yBuff,
  SyclBuffer** comms_bufferBuff, double* cg_alphas, double* cg_betas,
  double* cheby_alphas, double* cheby_betas, cl::sycl::queue** device_queue)
{
    delete(cg_alphas);
    delete(cg_betas);
    delete(cheby_alphas);
    delete(cheby_betas);

    delete(*device_queue);

    delete(*density0Buff);
    delete(*densityBuff);
    delete(*energy0Buff);
    delete(*energyBuff);
    delete(*uBuff);
    delete(*u0Buff);
    delete(*pBuff);
    delete(*rBuff);
    delete(*miBuff);
    delete(*wBuff);
    delete(*kxBuff);
    delete(*kyBuff);
    delete(*sdBuff);
    delete(*volumeBuff);
    delete(*x_areaBuff);
    delete(*y_areaBuff);
    delete(*cell_xBuff);
    delete(*cell_yBuff);
    delete(*cell_dxBuff);
    delete(*cell_dyBuff);
    delete(*vertex_dxBuff);
    delete(*vertex_dyBuff);
    delete(*vertex_xBuff);
    delete(*vertex_yBuff);
    delete(*comms_bufferBuff);
}
