#include <cuda_runtime.h>

#include "raytrace.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"
#include "camera.cuh"

int main() {
    hittable_list world;

    sphere* spheres = (sphere*)malloc(sizeof(sphere) * 2);
    spheres[0] = sphere(point3(0, 0, -1), 0.5);
    spheres[1] = sphere(point3(0, -100.5, -1), 100);

    camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width  = 400;

    cam.render(spheres);
    return 0;
}