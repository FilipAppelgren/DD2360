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
        
        particles[i].pos = add_float3(particles[i].pos, particles[i].vel);
    }

}

int main(int argc, char** argv)
{   
    int n_particles, n_iterations, n_threads;

    n_iterations = 1000;
    n_particles = 1000000;
    n_threads = 256;


    int grid_size = n_particles/n_threads;
    if (n_particles%n_threads != 0){
        grid_size++;
    }

    int bytes = sizeof(Particle) * n_particles;
   
    Particle *particles;
    cudaMallocManaged(&particles, bytes);

    for(int i = 0; i < n_particles; i++)
    {
        float3 random_vel = random_velocity();
        particles[i] = Particle(random_vel);
    }

    for(int i = 0; i < n_iterations; i++){
        timestep_update<<<grid_size, n_threads>>>(particles, n_particles);
    }
}
