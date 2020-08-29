#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <device_launch_parameters.h>

  //#include <stdio.h>
#include <iostream>


#include <stdlib.h>
#include <time.h>
#include <algorithm>
using namespace std;

#define n 64000000
#define BLOCKSIZE 512

__global__ void update(float* u, float* u_prev, int N, float dx, float dt, float c)
{
	// Each thread will load one element
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= N) { return; }
	u_prev[i] = u[i];
	if (i > 0)
	{
		u[i] = u_prev[i] - c * dt / dx * (u_prev[i] - u_prev[i-1]);
	}
}

__global__ void updateShared(float* u, float* u_prev, int N, float dx, float dt, float c)
{
	
	// Each thread will load one element
	int i = threadIdx.x;
	int I = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ float u_shared[BLOCKSIZE];
	if (I >= N) { return; }
	u_prev[I] = u[I];
	u_shared[i] = u[I];
	__syncthreads();
	if (i > 0 && i < BLOCKSIZE-1)
	{
		u[I] = u_shared[i] - c * dt / dx * (u_shared[i] - u_shared[i-1]);
	}
	
	else
	{
		u[I] = u_prev[I] - c * dt / dx * (u_prev[I] - u_prev[I-1]);
	}
	
	
}

/**
 * Host main routine
 */
int main(void)
{
	// Error code to check return values for CUDA calls
	//cudaError_t err = cudaSuccess;

	srand(time(NULL));
	

	// Print the vector length to be used, and compute its size
	
	size_t size = n * sizeof(float);
	

	// Allocate the host input vector A
	float* h_A = (float*)malloc(size);

	// Initialize the host input vectors
	for (int i = 0; i <  n; ++i)
	{
		h_A[i] = (float)(rand()%100);	
	}

	cout << "Data" << endl;
	for (int i = 0; i < 100; ++i)
	{
				cout << h_A[i] << ",";
	}
	cout << endl;
	// Allocate the device input vector A
	float* d_A = NULL;
	cudaMalloc((void**)& d_A, size);

	
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);





	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 512;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	cout << endl;
	clock_t start, endl, start_s, end_s;
	start = clock();
	update << <blocksPerGrid, threadsPerBlock >> > (d_A, d_A, n, .4, 0.2, 1.9);
	endl = clock();
	float* h_C = (float*)malloc(size);
	cudaMemcpy(h_C, d_A, size, cudaMemcpyDeviceToHost);


	cout << endl << "Result:" << endl;
	for (int i = 0; i < 100; i++)
		cout << " , " << h_C[i];

	cout << endl;
	cout << endl << "" << endl;
	cout << "Time needed:" << endl << (endl - start) << "ms";


	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	start_s = clock();
	updateShared << <blocksPerGrid, threadsPerBlock >> > (d_A, d_A, n, .4, 0.2, 1.9);
	end_s = clock();

	cudaMemcpy(h_C, d_A, size, cudaMemcpyDeviceToHost);


	cout << endl << "Shared Memory Result:" << endl;
	for (int i = 0; i < 100; i++)
		cout << " , " << h_C[i];

	cout << endl << "Break";
	cout <<  "Time needed:" << endl << (end_s - start_s) << "ms";
	// Free host memory
	free(h_A);

	free(h_C);

	printf("Done\n");
	return 0;
}