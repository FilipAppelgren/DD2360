#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <random>
#include <chrono>
#include <iostream>


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

    // Update velocity
    // particles[thread].vel = particles[thread].vel + particles[thread].vel; 

    // Update position
    particles[thread].pos = particles[thread].pos + particles[thread].vel;
    /*printf("Thread %d Coordinate X %f \n", thread, particles[thread].pos.x);
    printf("Thread %d Coordinate Y %f \n", thread, particles[thread].pos.y);
    printf("Thread %d Coordinate Z %f \n", thread, particles[thread].pos.z);*/
      
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

    if(argc == 4){
        n_iterations = std::stoi(argv[1]);
        n_particles = std::stoi(argv[2]);
        n_threads = std::stoi(argv[3]);
    } else {
        return 0;
    }

    int grid_size = n_particles/n_threads;
    if (n_particles%n_threads != 0){
        grid_size++;
    }

    if(n_threads > 1024)
        n_threads = 1024;

    int bytes = sizeof(Particle) * n_particles;
   
    // Allocate particle array on host
    Particle *particles, *gpu_particles, *cpu_particles;

    cudaMallocHost(&particles, sizeof(Particle) * n_particles);
    cudaMallocHost(&gpu_particles, sizeof(Particle) * n_particles);
    cudaMallocHost(&cpu_particles, sizeof(Particle) * n_particles);
    
    
    /*Particle particles[n_particles];
    Particle gpu_particles[n_particles];
    Particle cpu_particles[n_particles];*/
    
    // Initiate particles
    for(int i = 0; i < n_particles; i++)
    {
        float3 random_vel = random_velocity();
        particles[i] = Particle(random_vel);
    }
    memcpy(cpu_particles, particles, bytes);

    // Allocate on device
    
    int num_streams = 4;
    Particle *batches[num_streams];

    cudaStream_t streams[num_streams];

    int batch_size = bytes / num_streams;
    int batch_stride = n_particles / num_streams;
    for(int j = 0; j < n_iterations; j++){

        for (int i = 0; i < num_streams; i++) {

            cudaStreamCreate(&streams[i]);
            cudaMalloc(&batches[i], batch_size);

            int batch_number = batch_stride * i;

            cudaMemcpyAsync(batches[i], &gpu_particles[batch_number], batch_size, cudaMemcpyHostToDevice, streams[i]);
            timestep_update<<<grid_size, n_threads, 0, streams[i]>>>(batches[i], n_particles);
            cudaMemcpyAsync(&gpu_particles[batch_number], batches[i], batch_size, cudaMemcpyDeviceToHost, streams[i]);

            }
            //auto gpu_start = std::chrono::steady_clock::now();
            //auto gpu_end = std::chrono::steady_clock::now();
            //std::chrono::duration<double> elapsed_gpu_seconds = gpu_end-gpu_start;
        cudaDeviceSynchronize();
    }

    // Move particles from host to device


    // Update timestep



    auto cpu_start = std::chrono::steady_clock::now();
    for(int i = 0; i < n_iterations; i++){
        timestep_update_cpu(cpu_particles, n_particles);
    }

    auto cpu_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_cpu_seconds = cpu_end-cpu_start;


    // Error calculation, position

    float error = 0;
    
    for(int i = 0; i < n_particles; i++){
        //cpu_particles[i].print_particle();
        //printf("\n");
        //gpu_particles[i].print_particle();
        error += mag_float3(sub_float3(cpu_particles[i].pos, gpu_particles[i].pos));
    }
    error = error/n_particles;

    // Threads, iterations, particles, GPU time, CPU time
    
    //std::cout << n_threads << "," << n_iterations << "," << n_particles << "," << elapsed_gpu_seconds.count() << "," << elapsed_cpu_seconds.count() << "," << error << std::endl;
    //printf("%d,%d,%d,%f,%f\n", n_threads, n_iterations, n_particles, elapsed_gpu_seconds, elapsed_cpu_seconds);
}
