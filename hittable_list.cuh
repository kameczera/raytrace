#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.cuh"
#include "raytrace.cuh"
#include "sphere.cuh"
#include <vector>

class hittable_list {
    public:
        sphere* spheres;
        int object_count;

        __host__ __device__ hittable_list() : spheres(nullptr), object_count(0) {}
        __host__ __device__ hittable_list(sphere* spheres, int count) : spheres(spheres), object_count(count) {}

        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) {
            hit_record temp_rec;
            bool hit_anything = false;
            double closest_so_far = ray_t.max;

            for (int i = 0; i < object_count; i++) {
                if(spheres[i].hit_sphere(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }
};

#endif