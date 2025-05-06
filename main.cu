#include <cuda_runtime.h>

#include "raytrace.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"
#include "camera.cuh"
#include "material.cuh"

int main() {
    hittable_list world;
    int len_spheres = 4;
    sphere* spheres = (sphere*)malloc(sizeof(sphere) * len_spheres);
    spheres[0] = sphere(point3(0, -100.5, -1), 100, LAMBERTIAN, color(0.8, 0.8, 0.0), 0, 0);
    spheres[1] = sphere(point3(0, 0, -1.2), 0.5, LAMBERTIAN, color(0.1, 0.2, 0.5), 0, 0);
    spheres[2] = sphere(point3(-1.0, 0, -1.0), 0.5, DIELECTRIC, color(0.8, 0.8, 0.8), 0.3, 1.50);
    spheres[3] = sphere(point3(1.0, 0, -1.0), 0.5, METAL, color(0.8, 0.6, 0.2), 1.0, 0);

    camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width  = 400;

    cam.render(spheres, len_spheres);
    return 0;
}