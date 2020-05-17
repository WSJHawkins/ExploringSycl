#include <iostream>
#include <CL/sycl.hpp>

using namespace cl::sycl;
typedef buffer<double, 1> SyclBuffer;
unsigned long M = 2048;
queue* device_queue;

void initaliseStuff(queue** funcQueue,SyclBuffer* density);
void kernelBit(SyclBuffer& vertex_xBuff,SyclBuffer& aBuff, queue& funcQueue);

int main(void) {

  //Create Generic Buffer
  SyclBuffer densityBuff;

  // declare host arrays
  double *Ahost = new double[M];

  //Create a scope for SYCL
  {
	  
	  // Init arrays
	  for (int i = 0; i < M; ++i) Ahost[i] = 1.0;

	  initaliseStuff(&device_queue, &densityBuff);

	  // Creating 1D buffers for matrices which are bound to host arrays
	  SyclBuffer a{Ahost, range<1>{M}};

	  kernelBit(*density,a,*device_queue);

  }
  // Close the SYCL scope, the buffer destructor will write result back to Ahost.

  std::cout << "C[0]=" << Ahost[0] << std::endl;
  std::cout << "C[12]=" << Ahost[12] << std::endl;

}

void initaliseStuff(queue& funcQueue,SyclBuffer* densityBuff){
	*device_queue = new queue(cl::sycl::default_selector{});
    *densityBuff     = new SyclBuffer{range<1>{(size_t)M}};
}


void kernelBit(SyclBuffer& densityBuff,SyclBuffer& aBuff, queue& device_queue){

  device_queue.submit([&](handler &cgh){
  // Read from a and b, write to c
    auto aACC         = aBuff.get_access<access::mode::read_write>(cgh);
    auto densityAcc   = densityBuff.get_access<access::mode::discard_write>(cgh);

    // Executing kernel
    cgh.parallel_for<class MatrixMult>(range<1>{M}, [=](id<1> index){
      a[index] = densityAcc[index] * 2;
    });
	
  });
}

