#include <curand_kernel.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>

// #define NUM_ITER 10000000000
//#define grid_size 1
// #define BLOCK_SIZE 1
#define MAX_THREADS 2048*12

__global__
void calc_pi(uint64_t *counts, int iterations, int block_size)
{
    extern __shared__ int counter[];
    counter[threadIdx.x] = 0;
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int seed = id;

    curandState_t rng;
	curand_init(clock64(), seed, 0, &rng);
    float x, y, z;

    for(int i = 0; i < iterations; i++){
        x = curand_uniform(&rng);
        y = curand_uniform(&rng);
        z = sqrt((x*x) + (y*y));
        if(z <= 1){
            counter[threadIdx.x] += 1;
        }
    }
    __syncthreads();
    // Do we need to sync warps??
    if (threadIdx.x == 0) {
        counts[blockIdx.x] = 0;
		for (int i = 0; i < block_size; i++) {
			counts[blockIdx.x] += counter[i];
        }
        
	}
}


int main(int argc, char* argv[])
{
    uint64_t num_iter = std::stol(argv[1]);
    int block_size  = std::stol(argv[2]);
    int grid_size = MAX_THREADS/block_size;
    int n_thread_iterations = num_iter/(block_size * grid_size);

    printf("Total iterations %ld grid_size %d \n", num_iter, grid_size);

    uint64_t *h_total_counter, *d_total_counter;
    h_total_counter = new uint64_t[grid_size];
    cudaMalloc(&d_total_counter, sizeof(uint64_t) * grid_size);

    auto gpu_start = std::chrono::steady_clock::now();
    calc_pi<<<grid_size, block_size, block_size*sizeof(int)>>>(d_total_counter, n_thread_iterations, block_size);
    cudaMemcpy(h_total_counter, d_total_counter, sizeof(uint64_t) * grid_size, cudaMemcpyDeviceToHost);
    auto gpu_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_gpu_seconds = gpu_end-gpu_start;

    uint64_t total = 0;
    for(int i = 0; i < grid_size; i++){
        total += h_total_counter[i];
    }

    float pi = 4 * total/(float)num_iter;
    printf("PI: %f\n", pi);

    std::cout << "GPU Execution Time" << " " << elapsed_gpu_seconds.count() << std::endl;
}