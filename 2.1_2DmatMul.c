/**
 * Matrix Multiplication: C = A * B.
 *
 * This sample is a very basic sample that implements implements matrix multiplication
 * vector addition.
 * Matrices are stored in row-major order.
 */

#include <stdio.h>
#include <time.h>
#include <iostream>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "device_launch_parameters.h"

#define N 4

using namespace std;
/**
 * CUDA Kernel Device code
 *
 * Computes the matrix multiplication of A and B into C. The 3 vectors have the same
 * number of elements (N x N). 
 */
__global__ void
matMul(const float *A, const float *B, float* C, int N)
{
	int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int ROW = blockIdx.x;
	int COL = threadIdx.x;
	int ColSize = blockDim.x;

	float sum = 0;

	if (threadID < N * N) {
		for (int k = 0; k < N; k++)
		{
			sum += (A[ROW * ColSize + k] * B[k * ColSize + COL]);
		}
	}
	C[threadID] = sum;
}

/**
 * Host main routine
 */
int
main(void)
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	size_t size = (N * N) * sizeof(float);
	printf("[Matrix Multiplication of %d elements]\n", (N * N));

	// Variables to calculate time
	clock_t start, end;
	double CPU_time, GPU_time;
	
	// Allocate the host input matrix A & B
	float A[N][N];
	float B[N][N];

	// Nope notation:
	//float* A_1D = new float(N * N);
	//float* B_1D = new float(N * N);

	//Allocate the host input vector A
	float* h_A = (float*)malloc(sizeof(float) * N * N);

	//Allocate the host input vector B
	float* h_B = (float*)malloc(sizeof(float) * N * N);

	//Allocate the host input vector C
	float* h_C = (float*)malloc(sizeof(float) * N * N);

	// Randomly populate the matrix
	for (int r = 0; r < N; r++) {
		for (int c = 0; c < N; c++) {
			A[r][c] = (1.0) * (rand() % 100);
			h_A[r * N + c] = A[r][c];
			cout << h_A[r * N + c] << " ";
		}
		cout << endl;
	}

	cout << endl << endl;

	// Randomly populate the matrix
	for (int r = 0; r < N; r++) {
		for (int c = 0; c < N; c++) {
			B[r][c] = (1.0) * (rand() % 100);
			h_B[r * N + c] = B[r][c];
			cout << B[r][c] << " ";
		}
		cout << endl;
	}

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector A
	float* d_A = NULL;
	err = cudaMalloc((void**)& d_A, (sizeof(float) * N * N));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	float* d_B = NULL;
	err = cudaMalloc((void**)& d_B, (sizeof(float) * N * N));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	float* d_C = NULL;
	err = cudaMalloc((void**)& d_C, (sizeof(float) * N * N));


	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory

	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, (sizeof(float) * N * N), cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_B, h_B, (sizeof(float) * N * N), cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//Start time for GPU
	start = clock();

	threadsPerBlock = N;
	blocksPerGrid = N;
	printf("Third CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	matMul<<<blocksPerGrid, threadsPerBlock >>>(d_A, d_B, d_C, N);

	// End time for GPU
	end = clock();

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C, (sizeof(float) * N * N), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Start time for CPU
	start = clock();
	
	float C[N][N];
	cout << "CPU Result: " << endl;
	for (int r = 0; r < N; r++)
	{
		for (int c = 0; c < N; c++)
		{
			float sum = 0;
			for (int k = 0; k < N; k++)
			{
				sum += A[r][k] * B[k][c];
			}
			C[r][c] = sum;
			cout << C[r][c] << " ";
		}
		cout << endl;
	}

	// End time for CPU
	end = clock();

	// GPU Result:
	cout << "GPU Result: " << endl;
	for (int r = 0; r < N; r++)
	{
		for (int c = 0; c < N; c++)
		{
			cout << h_C[r * N + c] << " ";
		}
		cout << endl;
	}

    printf("Test PASSED\n");

	CPU_time = ((double)(end - start)) / CLOCKS_PER_SEC;
	GPU_time = ((double)(end - start)) / CLOCKS_PER_SEC;

	// Finally compare CPU and GPU time
	printf("CPU time: %f ms\n", CPU_time * 1000.0 );
	printf("GPU time: %f ms\n", GPU_time * 1000.0 );

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");

    return 0;
}

