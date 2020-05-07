#pragma once
#include "sycl_shared.hpp"

using namespace cl::sycl;

typedef buffer<double, 1>* FieldBufferType;

// Empty extension point
typedef struct ChunkExtension
{
    FieldBufferType comms_buffer;

} ChunkExtension;
