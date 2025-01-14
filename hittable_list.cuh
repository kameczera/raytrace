#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.cuh"
#include "raytrace.cuh"
#include <vector>

class hittable_list : public hittable {
    public:
        hittable** objects;
        int object_count;

        __host__ __device__ hittable_list() : objects(nullptr), object_count(0) {}
        __host__ __device__ hittable_list(hittable** objs, int count) : objects(objs), object_count(count) {}
        __host__ __device__ hittable_list(hittable** objs, int count) {
            
        }

        __host__ __device__ bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const override {
            hit_record temp_rec;
            bool hit_anything = false;
            double closest_so_far = ray_tmax;

            for (const hittable* object : objects) {
                if(object->hit(r, ray_tmin, closest_so_far, temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }
};

#endif