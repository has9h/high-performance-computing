#include <stdio.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <helper_functions.h>
#include <helper_cuda.h>
//#include "shfl_integral_image.cuh"
#include <iostream>
using namespace std;

__global__ void shfl_scan_test1(int* data, int width, int wSize, int* d_sz_partial)
{
	int warpSize = wSize;
	extern __shared__ int sums[];
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int lane_id = id % warpSize;
	// determine a warp_id within a block
	int warp_id = threadIdx.x / warpSize;
	int value = data[id];
	//int value1 = 0;
#pragma unroll

	
	unsigned int mask = 0xffffffff;
	for (int i = 1; i <= width; i = i * 2)
	{
		int n = __shfl_up_sync(mask, value, i, width);

		if (lane_id >= i) value += n;
		
		int kk = 100;
		int we = 100;
		int w7 = 100;
		__syncthreads();
	}
	int kk = 100;
	int we = 100;
	int w7 = 100;
	__syncthreads();
	__syncthreads();


	//data[id] = value;
	//Pick out 31st index and set the rest to 0:
	if(lane_id == (warpSize - 1))
	{
		sums[warp_id] = value;
		//data[id] = sums[warp_id];
	}
	else
	{
		//data[id] = 0;
	}

	int warp_sum;
	if (warp_id == 0 && lane_id < (blockDim.x / warpSize)){
		warp_sum = sums[lane_id];
		for (int i = 1; i <= (blockDim.x / warpSize); i = i * 2) {
			
			int n = __shfl_up_sync(mask, warp_sum, i, (blockDim.x / warpSize));
			
			if (lane_id >= i) warp_sum += n;
		
			int kk = 100;
			int we = 100;
			int w7 = 100;
			__syncthreads();
		}
		int kk = 100;
		int we = 100;
		int w7 = 100;
		__syncthreads();

		
	}

	__syncthreads();
	 kk = 100;
	  w7 = 100;
	 we = 100;
	
	if (threadIdx.x == (blockDim.x / warpSize) - 1) {
		int pos = (blockDim.x * (blockIdx.x + 1) - 1);
		//To insert in the first position: pos = (blockDim.x * blockIdx.x)
		data[pos] = warp_sum;
		d_sz_partial[blockIdx.x] = warp_sum;
	}
	else {
		data[id] = 0;
	}
	


	/*
		if ((threadIdx.x % warpSize) == (warpSize-1))
		{
			sums[warp_id] = value;
			data[id] = value;
		}
	*/
}

int main(int argc, char** argv)
{
	int* h_data, * h_result;
	int* d_data;
	int* d_sz_partial, *h_sz_partial;
	const int n_elements = 65536;
	int sz = sizeof(int) * n_elements;

	int cuda_device = 0;

	printf("Starting shfl_scan\n");

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	cuda_device = findCudaDevice(argc, (const char**)argv);

	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDevice(&cuda_device));
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

	checkCudaErrors(cudaMallocHost((void**)& h_data, sizeof(int) * n_elements));
	checkCudaErrors(cudaMallocHost((void**)& h_result, sizeof(int) * n_elements));

	//initialize data:
	printf("Computing Simple Sum test\n");
	printf("---------------------------------------------------\n");

	printf("Initialize test data [1, 1, 1...]\n");

	for (int i = 0; i < n_elements; i++)
	{
		h_data[i] = 1;
	}

	int blockSize = 256;
	int gridSize = (n_elements + 1) / blockSize;
	int nWarps = blockSize / 32;
	int shmem_sz = nWarps * sizeof(int);
	int partial_size = gridSize;
	int sz_partial = sizeof(int) * partial_size;
	//cout << "Shared mem " << shmem_sz;

	checkCudaErrors(cudaMalloc((void**)& d_data, sz));
	checkCudaErrors(cudaMalloc((void**)& d_sz_partial, sz_partial));

	checkCudaErrors(cudaMemcpy(d_data, h_data, sz, cudaMemcpyHostToDevice));
	shfl_scan_test1 << <gridSize, blockSize, shmem_sz >> > (d_data, 32, 32, d_sz_partial);
	
	checkCudaErrors(cudaMemcpy(h_result, d_data, sz, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMallocHost((void**)& h_sz_partial, sizeof(int) * sz_partial));
	checkCudaErrors(cudaMemcpy(h_sz_partial, d_sz_partial, sz_partial, cudaMemcpyDeviceToHost));

	cout << endl;
	for (int i = 0; i < (3 * blockSize); i++)
	{
		if (i % 32 == 0)
			cout << endl;

		if (i % blockSize == 0)
			cout << "newBlock:" << endl;

		cout << h_result[i] << ",";
	}

	for (int i = 0; i < gridSize; i++)
	{
		if (i % 32 == 0)
			cout << endl;

		if (i % blockSize == 0)
			cout << "newBlock:" << endl;

		cout << h_sz_partial[i] << ",";

	}

	cout << endl;

	//Running another kernel:
	blockSize = 64;
	int p_blockSize = min(partial_size, blockSize);
	int p_gridSize = (partial_size + 1) / p_blockSize;
	shfl_scan_test1 << <p_gridSize, p_blockSize, shmem_sz >> > (d_sz_partial, 32, 32, d_sz_partial);
	checkCudaErrors(cudaMemcpy(h_sz_partial, d_sz_partial, sz_partial, cudaMemcpyDeviceToHost));

	cout << endl;
	for (int i = 0; i < p_gridSize; i++)
	{
		if (i % 32 == 0)
			cout << endl;

		if (i % p_blockSize == 0)
			cout << "newBlock: ." << endl;

		cout << h_sz_partial[i] << ",";

		
	}

	shfl_scan_test1 << <1, p_gridSize, shmem_sz >>> (d_sz_partial,  p_gridSize, p_gridSize, d_sz_partial);
	checkCudaErrors(cudaMemcpy(h_sz_partial, d_sz_partial, sz_partial, cudaMemcpyDeviceToHost));
	cout << endl;
	for (int i = 0; i < 1; i++)
	{
		if (i % 32 == 0)
			cout << endl;

		if (i % p_blockSize == 0)
			cout << "newBlock:" << endl;

		cout << h_sz_partial[i] << ",";
	}
}