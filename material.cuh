#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "raytrace.cuh"
#include "hittable.cuh"

struct lambertian {
    color albedo;
    
    __host__ __device__ lambertian(const color& albedo) : albedo(albedo) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_state) {
        vec3 scatter_direction = rec.normal + random_unit_vector(local_state);
        
        if(scatter_direction.near_zero()) scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

struct metal {
    color albedo;
    
    __host__ __device__ metal(const color& albedo) : albedo(albedo) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_state) {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return true;
    }
};

template <typename T>
__device__ void scatter(T material, float3 ray_dir, float3 normal, float3* scattered_dir, float3* attenuation) {
    material.scatter(ray_dir, normal, *scattered_dir, *attenuation);
}

#endif
