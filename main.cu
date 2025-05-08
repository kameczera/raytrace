#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "raytrace.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"
#include "camera.cuh"
#include "material.cuh"

double random_double() {
    return rand() / (RAND_MAX + 1.0);
}

color random_color() {
    return color(random_double(), random_double(), random_double());
}

color random_color(double min, double max) {
    return color(
        min + (max - min) * random_double(),
        min + (max - min) * random_double(),
        min + (max - min) * random_double()
    );
}

int main() {
    srand(time(0));

    // Número estimado de esferas (ajustado conforme o grid)
    const int max_spheres = 500;
    sphere* spheres = (sphere*)malloc(sizeof(sphere) * max_spheres);
    int sphere_count = 0;

    // Ground
    spheres[sphere_count++] = sphere(point3(0, -1000, 0), 1000, LAMBERTIAN, color(0.5, 0.5, 0.5), 0, 0);

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            if (sphere_count >= max_spheres - 3) break; // segurança para não ultrapassar o limite

            double choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                if (choose_mat < 0.8) {
                    // Diffuse
                    color albedo = random_color() * random_color();
                    spheres[sphere_count++] = sphere(center, 0.2, LAMBERTIAN, albedo, 0, 0);
                } else if (choose_mat < 0.95) {
                    // Metal
                    color albedo = random_color(0.5, 1.0);
                    double fuzz = random_double() * 0.5;
                    spheres[sphere_count++] = sphere(center, 0.2, METAL, albedo, fuzz, 0);
                } else {
                    // Glass
                    spheres[sphere_count++] = sphere(center, 0.2, DIELECTRIC, color(0, 0, 0), 0, 1.5);
                }
            }
        }
    }

    // Três esferas grandes
    spheres[sphere_count++] = sphere(point3(0, 1, 0), 1.0, DIELECTRIC, color(0, 0, 0), 0, 1.5);
    spheres[sphere_count++] = sphere(point3(-4, 1, 0), 1.0, LAMBERTIAN, color(0.4, 0.2, 0.1), 0, 0);
    spheres[sphere_count++] = sphere(point3(4, 1, 0), 1.0, METAL, color(0.7, 0.6, 0.5), 0.0, 0);

    camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width  = 1200;
    cam.samples_per_pixel = 500;
    cam.max_depth = 50;

    cam.vfov = 20;
    cam.lookfrom = point3(13, 2, 3);
    cam.lookat   = point3(0, 0, 0);
    cam.vup      = vec3(0, 1, 0);

    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;

    cam.render(spheres, sphere_count);

    free(spheres);
    return 0;
}
