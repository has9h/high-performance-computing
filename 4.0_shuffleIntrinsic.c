#include <stdio.h>

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include "shfl_integral_image.cuh"
#include <iostream>
using namespace std;

__global__ void shfl_scan_test1(int* data, int width)
{
    //extern __shared__ int sums[];
    int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int lane_id = id % warpSize;
    // determine a warp_id within a block
    int warp_id = threadIdx.x / warpSize;
    int value = data[id];
    #pragma unroll

    unsigned int mask = 0xffffffff;
    int n = __shfl_up_sync(mask, value, i, width);
    if (lane_id >= i) value += n;
    data[id] = value;
    __syncthreads();
}

int main(int argc, char **argv)
{
    int *h_data , *h_result;
    int *d_data;
    const int n_elements = 65536;
    int sz = sizeof(int)*n_elements;
    int cuda_device = 0;

    printf("Starting shfl_scan\n");

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    cuda_device = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));
  

    checkCudaErrors(cudaMallocHost((void **)&h_data, sizeof(int)*n_elements));
    checkCudaErrors(cudaMallocHost((void **)&h_result, sizeof(int)*n_elements));

    //initialize data:
    printf("Computing Simple Sum test\n");
    printf("---------------------------------------------------\n");

    printf("Initialize test data [1, 1, 1...]\n");

    for (int i=0; i<n_elements; i++)
    {
        h_data[i] = 1;
    }

    int blockSize = 256;
    int gridSize = n_elements/blockSize;
    //int nWarps = blockSize/32;
    //int shmem_sz = nWarps * sizeof(int);
   

    checkCudaErrors(cudaMalloc((void **)&d_data, sz));
    checkCudaErrors(cudaMemcpy(d_data, h_data, sz, cudaMemcpyHostToDevice));
    shfl_scan_test1 << <gridSize, blockSize >> > (d_data, 32);
    checkCudaErrors(cudaMemcpy(h_result, d_data, sz, cudaMemcpyDeviceToHost));
    cout << endl;
	for (int i = 0; i < 100; i++)
		{
	if (i % 32 == 0)
	cout << endl;
	cout << h_result[i] << ",";

	}

    cout << endl;
}