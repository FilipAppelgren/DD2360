#!/bin/bash

# Compile the exercise and name it a.out

for n_iterations in 1000000 10000000 100000000 1000000000 10000000000 100000000000;
do
    touch "PI_fix$n_iterations.csv"
    #for n_threads in 16, 32, 64, 128, 256, 512, 1024;
    #do
    ./a.out $n_iterations 128 >> "PI_fix$n_iterations.csv"
    #done
done
