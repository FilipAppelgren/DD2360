#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <math.h>

#define ARRAY_SIZE 10000
#define BLOCK_SIZE 256

__global__
void saxpy(float *x, float *y, float a)
{
    int thread = blockIdx.x*blockDim.x + threadIdx.x;

    if (thread < ARRAY_SIZE){
        y[thread] = a * x[thread] + y[thread];
    }
}

void saxpyCPU(float *x, float *y, float a){

    for(int i = 0; i < ARRAY_SIZE; i++){
        y[i] = a * x[i] + y[i];
    }

}


int main()
{
    int nBlocks = (ARRAY_SIZE/BLOCK_SIZE);

    if (ARRAY_SIZE%BLOCK_SIZE != 0)  
        nBlocks++;

    int bytes = ARRAY_SIZE*sizeof(float);

    // CPU Allocation
    float *x = (float*)malloc(bytes);
    float *y = (float*)malloc(bytes);
    float *y_cpu = (float*)malloc(bytes);
    float A = std::rand() % 1000;

    // GPU Allocation
    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    for(int i = 0; i < ARRAY_SIZE; i++){
        x[i] = std::rand() % 1000;
        y[i] = std::rand() % 1000;
    }   
    memcpy(y_cpu, y, bytes);
        
    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);

    // GPU Saxpy
    saxpy<<<nBlocks, BLOCK_SIZE>>>(d_x, d_y, A);
    cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost);

    // CPU Saxypy
    saxpyCPU(x, y_cpu, A);

    float error = 0;

    printf("Comparing the output for each implementation…");
    for(int i = 0; i < ARRAY_SIZE; i++){
        error += abs(y[i] - y_cpu[i]);
    }
    printf("Total difference between GPU and CPU calculations: %.2f \n", error);


}