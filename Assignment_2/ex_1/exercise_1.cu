#include <stdlib.h>
#include <stdio.h>

__global__ void helloWorld()
{
    int thread = threadIdx.x;
    printf("Hello World! My threadId is %d \n", thread);
}


int main()
{
    helloWorld<<<1, 256>>>();
    cudaDeviceSynchronize();
}