#ifndef CAMERA_H
#define CAMERA_H

#include <chrono>

#include "raytrace.cuh"
#include "hittable_list.cuh"
#include "hittable.cuh"
#include "sphere.cuh"
#include "material.cuh"

__device__ color ray_color(const ray& r_in, hittable_list* world, int max_depth, curandState* local_state) {
    ray current_ray = r_in;
    color accumulated_attenuation = color(1.0, 1.0, 1.0);

    for (int depth = 0; depth < max_depth; ++depth) {
        hit_record rec;
        if (world->hit(current_ray, interval(0.001, infinity), rec)) {
            ray scattered;
            color attenuation;

            bool did_scatter = false;
            switch (rec.mat) {
                case LAMBERTIAN: {
                    lambertian mat(rec.col);
                    did_scatter = mat.scatter(current_ray, rec, attenuation, scattered, local_state);
                    break;
                }
                case METAL: {
                    metal mat(rec.col, rec.fuzz);
                    did_scatter = mat.scatter(current_ray, rec, attenuation, scattered, local_state);
                    break;
                }
                case DIELECTRIC:
                    dielectric mat(rec.refraction);
                    did_scatter = mat.scatter(current_ray, rec, attenuation, scattered, local_state);
                    break;
                default:
                    return color(0, 0, 0);
            }

            if (did_scatter) {
                accumulated_attenuation = accumulated_attenuation * attenuation;
                current_ray = scattered;
            } else {
                return color(0, 0, 0);
            }
        } else {
            vec3 unit_direction = unit_vector(current_ray.direction());
            double a = 0.5 * (unit_direction.y() + 1.0);
            color background = (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
            return accumulated_attenuation * background;
        }
    }

    return color(0, 0, 0);
}

__global__ void setup_rand_states(curandState* rand_state, unsigned long seed, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int id = row * width + col;
        curand_init(seed, id, 0, &rand_state[id]);
    }
}

__device__ point3 defocus_disk_sample(vec3 camera_center, curandState* rand_state, vec3 defocus_disk_u, vec3 defocus_disk_v) {
    auto p = random_in_unit_disk(rand_state);
    return camera_center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
}

__global__ void paint_gpu(int image_width, int image_height, hittable_list* world, int_color* d_colors, vec3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v, vec3 camera_center, int samples_per_pixel, curandState* rand_state, double defocus_angle, vec3 defocus_disk_u, vec3 defocus_disk_v, int max_depth) {
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
        vec3 offset = random(&local_state);
        vec3 pixel_sample = pixel00_loc + ((col + offset.x()) * pixel_delta_u) + ((row + offset.y()) * pixel_delta_v);
        vec3 ray_origin = (defocus_angle <= 0) ? camera_center : defocus_disk_sample(camera_center, rand_state, defocus_disk_u, defocus_disk_v);
        vec3 ray_direction = pixel_sample - ray_origin;
        ray r(ray_origin, ray_direction);
        pixel_color += ray_color(r, world, max_depth, &local_state);
    }

    pixel_color *= pixel_samples_scale;
    d_colors[row * image_width + col] = color_to_int(pixel_color);
    __syncthreads();
}

class camera {
    public:
        double aspect_ratio = 1.0;
        int image_width = 100;
        int samples_per_pixel = 100;
        double vfov = 90;
        point3 lookfrom = point3(0,0,0);
        point3 lookat = point3(0,0,-1);
        vec3 vup = vec3(0,1,0);
        double defocus_angle = 0;
        double focus_dist = 10;
        int max_depth = 50;

        void render(const sphere* spheres, int len_spheres) {
            initialize();
        
            auto start_total = std::chrono::high_resolution_clock::now();
        
            auto t1 = std::chrono::high_resolution_clock::now();
            std::clog << "\rAloccating memory on GPU..." << std::flush;
            sphere* d_spheres;
            cudaMalloc(&d_spheres, len_spheres * sizeof(sphere));
            cudaMemcpy(d_spheres, spheres, len_spheres * sizeof(sphere), cudaMemcpyHostToDevice);
            auto t2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration_alloc = t2 - t1;
        
            hittable_list h_world(d_spheres, len_spheres);
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
        
            auto t3 = std::chrono::high_resolution_clock::now();
            std::clog << "\rExecuting kernel...        " << std::flush;
            cudaMalloc(&d_rgb, rgb_size);
            paint_gpu<<< grid_dim, block_dim >>> (image_width, image_height, d_world, d_rgb, pixel00_loc, pixel_delta_u, pixel_delta_v, center, samples_per_pixel, d_rand_state, defocus_angle, defocus_disk_u, defocus_disk_v, max_depth);
            cudaDeviceSynchronize();
            auto t4 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration_kernel = t4 - t3;
        
            auto t5 = std::chrono::high_resolution_clock::now();
            std::clog << "\rCopying memory from GPU to CPU..." << std::flush;
            cudaMemcpy(h_rgb, d_rgb, rgb_size, cudaMemcpyDeviceToHost);
            auto t6 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration_copy = t6 - t5;
        
            auto t7 = std::chrono::high_resolution_clock::now();
            std::clog << "\rWriting Image .PPM format...     " << std::flush;
            std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
            for (int i = 0; i < image_width * image_height; i++) {
                write_color(std::cout, h_rgb[i]);
            }
            auto t8 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration_output = t8 - t7;
        
            free(h_rgb);
            cudaFree(d_rgb);
            cudaFree(d_spheres);
            cudaFree(d_world);
        
            auto end_total = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration_total = end_total - start_total;
        
            std::clog << "\n\n--- TEMPOS DE EXECUÇÃO ---\n";
            std::clog << "Alocação GPU:             " << duration_alloc.count() << " s\n";
            std::clog << "Execução kernel:          " << duration_kernel.count() << " s\n";
            std::clog << "Cópia GPU -> CPU:         " << duration_copy.count() << " s\n";
            std::clog << "Escrita imagem:           " << duration_output.count() << " s\n";
            std::clog << "Tempo total:              " << duration_total.count() << " s\n";
        }

    private:
        int image_height;
        double pixel_samples_scale;
        point3 center;
        point3 pixel00_loc;
        vec3 pixel_delta_u;
        vec3 pixel_delta_v;
        vec3 u, v, w;
        vec3   defocus_disk_u;
        vec3   defocus_disk_v; 

        __host__ void initialize() {
            image_height = int(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            pixel_samples_scale = 1.0 / samples_per_pixel;

            center = lookfrom;

            double theta = degrees_to_radians(vfov);
            double h = std::tan(theta/2);
            double viewport_height = 2 * h * focus_dist;
            double viewport_width = viewport_height * (double(image_width) / image_height);

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            vec3 viewport_u = viewport_width * u;
            vec3 viewport_v = viewport_width * -v;
            
            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            vec3 viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
            pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
            double defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;
        }
};

#endif