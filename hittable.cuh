#ifndef HITTABLE_H
#define HITTABLE_H

#include "raytrace.cuh"

class hit_record {
    public:
        point3 p;
        vec3 normal;
        double t;
        bool front_face;

        __host__ __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
            front_face = dot(r.direction(), outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
        }
};

#endif