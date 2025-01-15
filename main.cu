#include <cuda_runtime.h>

#include "raytrace.cuh"
#include "hittable_list.cuh"
#include "sphere.cuh"

__host__ __device__ color ray_color(const ray& r, hittable_list* world) {
    hit_record rec;
    if(world->hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + color(1, 1, 1));
    }

    vec3 unit_direction = unit_vector(r.direction());
    double a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

__global__ void paint_gpu(int image_width, int image_height, hittable_list* world, int_color* d_colors, vec3 pixel00_loc, vec3 pixel_delta_u, vec3 pixel_delta_v, vec3 camera_center) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < image_height && col < image_width) {
        vec3 pixel_center = pixel00_loc + (col * pixel_delta_u) + (row * pixel_delta_v);
        vec3 ray_direction = pixel_center - camera_center;
        ray r(camera_center, ray_direction);
        int_color pixel_color = color_to_int(ray_color(r, world));
        d_colors[row * image_width + col] = pixel_color;
    }
}

int main() {

    double aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    sphere* h_spheres = (sphere*)malloc(sizeof(sphere) * 2);
    h_spheres[0] = sphere(point3(0, 0, -1), 0.5);
    h_spheres[1] = sphere(point3(0, -100.5, -1), 100);

    sphere* d_spheres;
    cudaMalloc(&d_spheres, 2 * sizeof(sphere));
    cudaMemcpy(d_spheres, h_spheres, 2 * sizeof(sphere), cudaMemcpyHostToDevice);

    free(h_spheres);

    hittable_list h_world(d_spheres, 2);

    hittable_list* d_world;
    cudaMalloc(&d_world, sizeof(hittable_list));
    cudaMemcpy(d_world, &h_world, sizeof(hittable_list), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro CUDA: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

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

    std::clog << "\rExecuting kernel...        " << std::flush;
    paint_gpu<<< grid_dim, block_dim >>> (image_width, image_height, d_world, d_rgb, pixel00_loc, pixel_delta_u, pixel_delta_v, camera_center);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch paint_gpu kernel: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }


    std::clog << "\rCopying memory from GPU to CPU..." << std::flush;
    cudaMemcpy(h_rgb, d_rgb, rgb_size, cudaMemcpyDeviceToHost);
    err = cudaMemcpy(h_rgb, d_rgb, rgb_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy memory from device to host: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }


    std::clog << "\rWriting Image .PPM format...     " << std::flush;
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for(int i = 0;  i < image_width * image_height; i++) {
        std::cout << h_rgb[i] << '\n';
    }

    free(h_rgb);
    cudaFree(d_rgb);
    cudaFree(d_spheres);
    cudaFree(d_world);

    return 0;
}