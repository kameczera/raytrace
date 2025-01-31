#ifndef CAMERA_H
#define CAMERA_H

#include "raytrace.cuh"
#include "hittable_list.cuh"
#include "hittable.cuh"
#include "sphere.cuh"

#include <curand_kernel.h>
#include <cuda_runtime.h>

__device__ color ray_color(const ray& r, hittable_list* world) {
    hit_record rec;
    if(world->hit(r, interval(0, infinity), rec)) {
        return 0.5 * (rec.normal + color(1, 1, 1));
    }
    vec3 unit_direction = unit_vector(r.direction());
    double a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

__global__ void setup_rand_states(curandState* rand_state, unsigned long seed, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int id = row * width + col;
        curand_init(seed, id, 0, &rand_state[id]);
    }
}

__global__ void paint_gpu(int image_width, int image_height, hittable_list* world, int_color* d_colors, vec3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v, vec3 camera_center, int samples_per_pixel, curandState* rand_state) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= image_height || col >= image_width) return;

    curandState local_state = rand_state[row * image_width + col];

    double pixel_samples_scale = 1.0 / samples_per_pixel;
    vec3 pixel_center = pixel00_loc + (col * pixel_delta_u) + (row * pixel_delta_v);
    vec3 ray_direction = pixel_center - camera_center;
    ray r(camera_center, ray_direction);
    color pixel_color;

    for (int s = 0; s < samples_per_pixel; ++s) {
        vec3 offset(curand_uniform(&local_state), curand_uniform(&local_state), curand_uniform(&local_state));
        vec3 pixel_sample = pixel00_loc + ((col + offset.x()) * pixel_delta_u) + ((row + offset.y()) * pixel_delta_v);
        vec3 ray_origin = camera_center;
        vec3 ray_direction = pixel_sample - ray_origin;
        ray r(ray_origin, ray_direction);
        pixel_color += ray_color(r, world);
    }

    pixel_color *= pixel_samples_scale;
    d_colors[row * image_width + col] = color_to_int(pixel_color);
}

class camera {
    public:
        double aspect_ratio = 1.0;
        int image_width = 100;
        int samples_per_pixel = 10;

        void render(const sphere* spheres) {
            initialize();

            sphere* d_spheres;
            cudaMalloc(&d_spheres, 2 * sizeof(sphere));
            cudaMemcpy(d_spheres, spheres, 2 * sizeof(sphere), cudaMemcpyHostToDevice);

            hittable_list h_world(d_spheres, 2);

            hittable_list* d_world;
            cudaMalloc(&d_world, sizeof(hittable_list));
            cudaMemcpy(d_world, &h_world, sizeof(hittable_list), cudaMemcpyHostToDevice);

            curandState* d_rand_state;
            cudaMalloc((void**)&d_rand_state, image_width * image_height * sizeof(curandState));

            dim3 blockDim(32, 32);
            dim3 gridDim((image_width + blockDim.x - 1) / blockDim.x, (image_height + blockDim.y - 1) / blockDim.y);
            setup_rand_states<<<gridDim, blockDim>>>(d_rand_state, 1234, image_width, image_height);

            dim3 block_dim(16,16,1);
            dim3 grid_dim((image_width + block_dim.x - 1) / block_dim.x, (image_height + block_dim.y - 1) / block_dim.y);

            size_t rgb_size = sizeof(color) * image_width * image_height;
            int_color* d_rgb;
            int_color* h_rgb = (int_color*)malloc(rgb_size);
            std::clog << "\rAloccating memory on GPU..." << std::flush;
            cudaMalloc(&d_rgb, rgb_size);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate memory: " << cudaGetErrorString(err) << std::endl;
            }

            std::clog << "\rExecuting kernel...        " << std::flush;
            paint_gpu<<< grid_dim, block_dim >>> (image_width, image_height, d_world, d_rgb, pixel00_loc, pixel_delta_u, pixel_delta_v, center, samples_per_pixel, d_rand_state);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "Failed to launch paint_gpu kernel: " << cudaGetErrorString(err) << std::endl;
            }


            std::clog << "\rCopying memory from GPU to CPU..." << std::flush;
            cudaMemcpy(h_rgb, d_rgb, rgb_size, cudaMemcpyDeviceToHost);
            err = cudaMemcpy(h_rgb, d_rgb, rgb_size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cerr << "Failed to copy memory from device to host: " << cudaGetErrorString(err) << std::endl;
            }


            std::clog << "\rWriting Image .PPM format...     " << std::flush;

            std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
            for (int i = 0; i < image_width * image_height; i++) {
                write_color(std::cout, h_rgb[i]);
            }

            free(h_rgb);
            cudaFree(d_rgb);
            cudaFree(d_spheres);
            cudaFree(d_world);
        }

    private:
        int image_height;
        double pixel_samples_scale;
        point3 center;
        point3 pixel00_loc;
        vec3 pixel_delta_u;
        vec3 pixel_delta_v;

        __host__ void initialize() {
            image_height = int(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            pixel_samples_scale = 1.0 / samples_per_pixel;

            double focal_length = 1.0;
            double viewport_height = 2.0;
            double viewport_width = viewport_height * (double(image_width) / image_height);
            center = point3(0, 0, 0);

            vec3 viewport_u = vec3(viewport_width, 0, 0);
            vec3 viewport_v = vec3(0,-viewport_height, 0);

            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            vec3 viewport_upper_left = center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
            pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
        }
};

#endif