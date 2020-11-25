#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <random>
#include <iostream>
#include <curand_kernel.h>
#include <ctime>


class Particle
{
    public:

        float3 pos = make_float3(0,0,0);
        float3 vel = make_float3(1,1,1);

        Particle() {}

        Particle(float3 velocity){
            vel = velocity;
        }
        
        void print_particle() {
            printf("position (%f,%f,%f) \n", pos.x, pos.y, pos.z);
            printf("velocity (%f,%f,%f) \n", vel.x, vel.y, vel.z);
        }
    
};

__device__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 add_float3(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 sub_float3(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float mag_float3(const float3 &a)
{
    return abs(a.x) + abs(a.y) + abs(a.z);
}


__global__
void timestep_update(Particle *particles, int n_particles)
{

    int thread = blockIdx.x*blockDim.x + threadIdx.x;
    // Avoid index out of bounds
    if(thread > n_particles - 1){
        return;
    }

    
    // particles[thread].vel = particles[thread].vel + particles[thread].vel; 

    // Update position
    particles[thread].pos = particles[thread].pos + particles[thread].vel;
    /*printf("Thread %d Coordinate X %f \n", thread, particles[thread].pos.x);
    printf("Thread %d Coordinate Y %f \n", thread, particles[thread].pos.y);
    printf("Thread %d Coordinate Z %f \n", thread, particles[thread].pos.z);*/
      
}

__global__
void dummy_kernel(float *d_out, int N) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init((unsigned long long)clock() + i, 0, 0, &state);
    
    for(int j = 0; j < N; j++){
        d_out[j] = curand_uniform_double(&state);
    }

}

float3 random_velocity()
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(-10, 10);
    
    auto x = uni(rng);
    auto y = uni(rng);
    auto z = uni(rng);

    return make_float3(x,y,z);
}

void timestep_update_cpu(Particle *particles, int n_particles){

    for(int i = 0; i < n_particles; i++){
        //particles[i].vel = add_float3(particles[i].vel, particles[i].vel); 
        particles[i].pos = add_float3(particles[i].pos, particles[i].vel);
    }

}

int main(int argc, char** argv)
{   
    int n_particles, n_iterations, n_threads;

    n_iterations = 1000;
    n_particles = 1000000;
    n_threads = 256;

    //int grid_size = n_particles/n_threads;
    int grid_size = 10;
    if (n_particles%n_threads != 0){
        grid_size++;
    }

    int bytes = sizeof(Particle) * n_particles;
   
    // Allocate particle array on host
    Particle *particles, *gpu_particles;

    cudaMallocHost(&particles, sizeof(Particle) * n_particles);
    cudaMallocHost(&gpu_particles, sizeof(Particle) * n_particles);
    
    // Initiate particles
    for(int i = 0; i < n_particles; i++)
    {
        float3 random_vel = random_velocity();
        particles[i] = Particle(random_vel);
    }

    // Allocate on device
    
    int num_streams = 4;
    Particle *batches[num_streams];

    cudaStream_t streams[num_streams];

    int batch_size = bytes / num_streams;
    int batch_stride = n_particles / num_streams;

    // Create streams
    for(int i = 0; i < num_streams; i++){
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&batches[i], batch_size);
    }

    // Dummy variables
    float *d_out;
    int N = 1000;
    cudaMalloc((void**)&d_out, N * sizeof(float));

    for(int j = 0; j < n_iterations; j++){
        
        for (int i = 0; i < num_streams; i++) {
            int batch_number = batch_stride * i;

            cudaMemcpyAsync(batches[i], &gpu_particles[batch_number], batch_size, cudaMemcpyHostToDevice, streams[i]);
            timestep_update<<<grid_size, n_threads, 0, streams[i]>>>(batches[i], n_particles);
            dummy_kernel<<<grid_size, n_threads, 0, streams[i]>>>(d_out, N);
            cudaMemcpyAsync(&gpu_particles[batch_number], batches[i], batch_size, cudaMemcpyDeviceToHost, streams[i]);
            }

        cudaDeviceSynchronize();
    }

    for(int i = 0; i < num_streams; i++){
        cudaStreamDestroy(streams[i]);
    }

}
