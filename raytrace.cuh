#ifndef RAYTRACE_H
#define RAYTRACE_H

#include <cmath>
#include <limits>
#include <memory>
#include <iostream>
#include <cstdlib>

#include <curand_kernel.h>
#include <cuda_runtime.h>

using std::make_shared;
using std::shared_ptr;

__device__ const double infinity = std::numeric_limits<double>::infinity();
__device__ const double pi = 3.1415926535897932385;

__host__ __device__ inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

#include "color.cuh"
#include "ray.cuh"
#include "vec3.cuh"
#include "interval.cuh"

#endif