#ifndef COLOR_H
#define COLOR_H

#include "vec3.cuh"

#include <iostream>

using color = vec3;

class int_color {
    public:
        int e[3];
        __host__ __device__ int_color() : e{0,0,0} {}
        __host__ __device__ int_color(int e0, int e1, int e2) : e{e0, e1, e2} {}
};

__host__ inline std::ostream& operator<<(std::ostream& out, const int_color& color) {
    return out << color.e[0] << ' ' << color.e[1] << ' ' << color.e[2];
}

__host__ __device__ int_color color_to_int(const color& pixel_color) {
    double r = pixel_color.x();
    double g = pixel_color.y();
    double b = pixel_color.z();

    int rbyte = int(255.999 * r);
    int gbyte = int(255.999 * g);
    int bbyte = int(255.999 * b);

    return int_color(rbyte, gbyte, bbyte);
}

#endif