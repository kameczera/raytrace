#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.cuh"
#include "raytrace.cuh"
#include "material.cuh"

class sphere {
    public:
        __host__ __device__ sphere() : center(point3(0, 0, 0)), radius(0), mat(LAMBERTIAN) {}
        __host__ __device__ sphere(const point3& center, double radius, material_type mat, color col, double fuzz, double refraction) : center(center), radius(std::fmax(0, radius)), mat(mat), col(col), fuzz(fuzz), refraction(refraction) {}

        __device__ bool hit_sphere(const ray& r, interval ray_t, hit_record& rec) {
            vec3 oc = center - r.origin();
            double a = r.direction().length_squared();
            double h = dot(r.direction(), oc);
            double c = oc.length_squared() - radius * radius;

            double discriminant = h * h - a * c;
            if(discriminant < 0) return false;

            double sqrtd = std::sqrt(discriminant);

            double root = (h - sqrtd) / a;
            if(!ray_t.surrounds(root)) {
                root = (h + sqrtd) / a;
                if(!ray_t.surrounds(root)) return false;
            }
            rec.col = col;
            if(mat == METAL) rec.fuzz = fuzz;
            else if(mat == DIELECTRIC) rec.refraction = refraction;
            rec.t = root;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat = mat;
            return true;
        }

    private:
        point3 center;
        double radius;
        color col;
        material_type mat;
        double refraction = 0;
        double fuzz = 0;
};

#endif