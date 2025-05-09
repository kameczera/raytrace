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
    double fuzz;
    __host__ __device__ metal(const color& albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_state) {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + (fuzz * random_unit_vector(local_state));
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
};

struct dielectric {
    double refraction_index;

    __host__ __device__ dielectric(double refraction_index) : refraction_index(refraction_index) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_state) {
        attenuation = color(1.0, 1.0, 1.0);
        double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
        bool cannot_refract = ri * sin_theta > 1.0;
        vec3 direction;

        if(cannot_refract ||(reflectance(cos_theta, ri) > random_double(local_state))) direction = refract(unit_direction, rec.normal, ri);
        else direction = refract(unit_direction, rec.normal, ri);
        scattered = ray(rec.p, direction);
        return true;
    }

    __device__ double reflectance(double cosine, double refraction_index) {
        double r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * std::pow((1 - cosine), 5);
    }
};

template <typename T>
__device__ void scatter(T material, float3 ray_dir, float3 normal, float3* scattered_dir, float3* attenuation) {
    material.scatter(ray_dir, normal, *scattered_dir, *attenuation);
}

#endif
