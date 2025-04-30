#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "raytrace.cuh"

// Declaração forward para hit_record
struct hit_record;

enum material_type { LAMBERTIAN, METAL, NONE };

struct lambertian {
    color albedo;
    
    __host__ __device__ lambertian(const color& albedo) : albedo(albedo) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_state) const;
};

struct metal {
    color albedo;
    
    __host__ __device__ metal(const color& albedo) : albedo(albedo) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_state) const;
};

template <typename T>
__device__ void scatter(T material, float3 ray_dir, float3 normal, float3* scattered_dir, float3* attenuation) {
    material.scatter(ray_dir, normal, *scattered_dir, *attenuation);
}

#endif