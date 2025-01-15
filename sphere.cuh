#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.cuh"
#include "raytrace.cuh"

class sphere{
    public:
        __host__ __device__ sphere() : center(point3(0, 0, 0)), radius(0) {}
        __host__ __device__ sphere(const point3& center, double radius) : center(center), radius(std::fmax(0, radius)) {}

        __host__ __device__ bool hit_sphere(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) {
            vec3 oc = center - r.origin();
            double a = r.direction().length_squared();
            double h = dot(r.direction(), oc);
            double c = oc.length_squared() - radius * radius;

            double discriminant = h * h - a * c;
            if(discriminant < 0) return false;

            double sqrtd = std::sqrt(discriminant);

            double root = (h - sqrtd) / a;
            if(root <= ray_tmin || ray_tmax <= root) {
                root = (h + sqrtd) / a;
                if(root <= ray_tmin || ray_tmax <= root) return false;
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            return true;
        }

    private:
        point3 center;
        double radius;
};

#endif