#!/bin/bash

# Compile the particle simulation and name it a.out

iterations=100000

for n_particles in 10 100 1000 10000 100000; 
do
    touch "test_particles$n_particles.csv"
    echo "threads,iterations,particles,gpu,cpu,error" >> "test_particles$n_particles.csv"
    for n_threads in 32 64 128 256 512 1024; 
    do
        ./a.out $iterations $n_particles $n_threads >> "test_particles$n_particles.csv"
    done
done