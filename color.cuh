#ifndef COLOR_H
#define COLOR_H

#include "vec3.cuh"

#include "raytrace.cuh"
#include "interval.cuh"

using color = vec3;

class int_color {
    public:
        int e[3];
        __host__ __device__ int_color() : e{0,0,0} {}
        __host__ __device__ int_color(int e0, int e1, int e2) : e{e0, e1, e2} {}
};

__device__ inline color operator+=(color& color, const double t) {
    color.e[0] += t;
    color.e[1] += t;
    color.e[2] += t;
    return color;
}

__host__ inline std::ostream& operator<<(std::ostream& out, const int_color& color) {
    return out << color.e[0] << ' ' << color.e[1] << ' ' << color.e[2];
}

__device__ inline int_color operator*(int_color& int_color, const double t) {
    int_color.e[0] = int(int_color.e[0] * t);
    int_color.e[1] = int(int_color.e[1] * t);
    int_color.e[2] = int(int_color.e[2] * t);
    return int_color;
}

__device__ inline int_color& operator+=(int_color& ic1, const int_color ic2) {
    ic1.e[0] += ic2.e[0];
    ic1.e[1] += ic2.e[1];
    ic1.e[2] += ic2.e[2];
    return ic1;
}

__device__ int_color color_to_int(const color& pixel_color) {
    double r = pixel_color.x();
    double g = pixel_color.y();
    double b = pixel_color.z();
    interval intensity(0.000, 0.999);
    int rbyte = int(256 * intensity.clamp(r));
    int gbyte = int(256 * intensity.clamp(g));
    int bbyte = int(256 * intensity.clamp(b));

    return int_color(rbyte, gbyte, bbyte);
}

__host__ void write_color(std::ostream& out, const int_color& pixel_color) {
    out << pixel_color.e[0] << ' ' << pixel_color.e[1] << ' ' << pixel_color.e[2] << '\n';
}

#endif