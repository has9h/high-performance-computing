#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <iostream>
#include <time.h>
#include <assert.h>
#include <cuda.h>
#include <math.h>
using namespace std;



#define THREADS 256 // 2^9
#define BLOCKS (int)pow(2, 10) // 2^15
#define NUM_VALS THREADS*BLOCKS


void print_elapsed(clock_t start, clock_t stop)
{
	double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
	printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
	return (float)(rand() %1000);
}

void array_print(float* arr, int length)
{
	int i;
	for (i = 0; i < length - 1; ++i) {
		
		if (arr[i+1] != arr[i]) printf("%1.0f ", arr[i]);
	}
	printf("\n\n\n");
}

void array_fill(float* arr, int length)
{
	srand(time(NULL));
	int i;
	for (i = 0; i < length; ++i) {
		arr[i] = random_float();

		if (i % 1000 == 0) 	srand(time(NULL));
		
	}
}

__global__ void bitonic_sort_step(float* dev_values, int j, int k)
{
	unsigned int i, ixj; /* Sorting partners: i and ixj */
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i ^ j;
	float temp;

	/* The threads with the lowest ids sort the array. */
	if ((ixj) > i) {
		if ((i & k) == 0) {
			/* Sort ascending */
			if (dev_values[i] > dev_values[ixj]) {
				/* exchange(i,ixj); */
				temp = dev_values[i];
				dev_values[i] = dev_values[ixj];
				dev_values[ixj] = temp;
			}
		}
		if ((i & k) != 0) {
			/* Sort descending */
			if (dev_values[i] < dev_values[ixj]) {
				/* exchange(i,ixj); */
				temp = dev_values[i];
				dev_values[i] = dev_values[ixj];
				dev_values[ixj] = temp;
			}
		}
	}
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float* values)
{
	float* dev_values;
	size_t size = NUM_VALS * sizeof(float);

	cudaMalloc((void**)& dev_values, size);
	cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

	dim3 blocks(BLOCKS, 1);    /* Number of blocks   */
	dim3 threads(THREADS, 1);  /* Number of threads  */

	int j, k;
	/* Major step */
	for (k = 2; k <= NUM_VALS; k <<= 1) {
		/* Minor step */
		for (j = k >> 1; j > 0; j = j >> 1) {
			bitonic_sort_step << <blocks, threads >> > (dev_values, j, k);
		}
		//array_print(values, 1000);

		
	}
	cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
	cudaFree(dev_values);
}

int main(int argc, char *argv[])
{
    // Initialization.  The shuffle intrinsic is not available on SM < 3.0
    // so waive the test if the hardware is not present.
    int cuda_device = 0;



    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    cuda_device = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // __shfl intrinsic needs SM 3.0 or higher
    if (deviceProp.major < 3)
    {
        printf("> __shfl() intrinsic requires device SM 3.0+\n");
        printf("> Waiving test.\n");
        exit(EXIT_WAIVED);
    }

	clock_t start, stop;
	float* values = (float*)malloc(NUM_VALS * sizeof(float));
	array_fill(values, NUM_VALS);
	//array_print(values, NUM_VALS);

	start = clock();
	bitonic_sort(values); /* Inplace */
	stop = clock();
	array_print(values, NUM_VALS);
	print_elapsed(start, stop);
}