#include <iostream>
#include <cuda_runtime.h>

#include "color.cuh"
#include "vec3.cuh"

__global__ void paint_gpu(int image_width, int image_height, color* d_colors) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < image_height && col < image_width) {
        color pixel_color = color(double(row) / (image_width - 1), double(col) / (image_height - 1), 0);
        d_colors[row * image_height + col] = pixel_color;
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
    int image_width = 256;
    int image_height = 256;

    dim3 block_dim(32,32,1);
    dim3 grid_dim((image_width + block_dim.x - 1) / block_dim.x, (image_height + block_dim.y - 1) / block_dim.y);

    size_t rgb_size = sizeof(int) * image_width * image_height * 3;
    color* d_rgb;
    color* h_rgb = (color*)malloc(rgb_size);
    std::clog << "\rAloccating memory on GPU..." << std::flush;
    cudaMalloc(&d_rgb, rgb_size);

    std::clog << "\rExecuting kernel..." << std::flush;
    paint_gpu<<< grid_dim, block_dim >>> (image_width, image_height, d_rgb);

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