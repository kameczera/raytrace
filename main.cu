#include <iostream>
#include <cuda_runtime.h>

#include "color.cuh"
#include "vec3.cuh"
#include "ray.cuh"


__host__ __device__ color ray_color(const ray& r) {
    vec3 unit_direction = unit_vector(r.direction());
    double a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

__global__ void paint_gpu(int image_width, int image_height, int_color* d_colors, vec3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v, vec3 camera_center) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < image_height && col < image_width) {
        vec3 pixel_center = pixel00_loc + (col * pixel_delta_u) + (row * pixel_delta_v);
        vec3 ray_direction = pixel_center - camera_center;
        ray r(camera_center, ray_direction);
        int_color pixel_color = color_to_int(ray_color(r));
        d_colors[row * image_width + col] = pixel_color;
    }
}

__host__ void paint_cpu(int image_width, int image_height, int* rgb) {
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            auto r = double(i) / (image_width-1);
            auto g = double(j) / (image_height-1);
            auto b = 0.0;

            int ir = int(255.999 * r);
            int ig = int(255.999 * g);
            int ib = int(255.999 * b);

            rgb[i * j + i] = ir;
            rgb[i * j + i + 1] = ig;
            rgb[i * j + i + 2] = ib;
        }
    }
}

int main() {

    double aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    double focal_length = 1.0;
    double viewport_height = 2.0;
    double viewport_width = viewport_height * (double(image_width) / image_height);
    point3 camera_center = point3(0, 0, 0);

    vec3 viewport_u = vec3(viewport_width, 0, 0);
    vec3 viewport_v = vec3(0,-viewport_height, 0);

    vec3 pixel_delta_u = viewport_u / image_width;
    vec3 pixel_delta_v = viewport_v / image_height;

    vec3 viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
    vec3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    
    dim3 block_dim(32,32,1);
    dim3 grid_dim((image_width + block_dim.x - 1) / block_dim.x, (image_height + block_dim.y - 1) / block_dim.y);

    size_t rgb_size = sizeof(color) * image_width * image_height;
    int_color* d_rgb;
    int_color* h_rgb = (int_color*)malloc(rgb_size);
    std::clog << "\rAloccating memory on GPU..." << std::flush;
    cudaMalloc(&d_rgb, rgb_size);

    std::clog << "\rExecuting kernel..." << std::flush;
    paint_gpu<<< grid_dim, block_dim >>> (image_width, image_height, d_rgb, pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center);

    std::clog << "\rCopying memory from GPU to CPU..." << std::flush;
    cudaMemcpy(h_rgb, d_rgb, rgb_size, cudaMemcpyDeviceToHost);

    std::clog << "\rWriting Image .PPM format...     " << std::flush;
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for(int i = 0;  i < image_width * image_height; i++) {
        std::cout << h_rgb[i] << '\n';
    }

    free(h_rgb);
    cudaFree(d_rgb);

    return 0;
}