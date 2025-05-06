#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include "raytrace.cuh"

enum material_type { LAMBERTIAN, METAL, DIELECTRIC, NONE };

class hit_record {
public:
    point3 p;
    color col;
    double fuzz;
    vec3 normal;
    material_type mat;
    double t;
    bool front_face;
    double refraction;

    __host__ __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

#endif
