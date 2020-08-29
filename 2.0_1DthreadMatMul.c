/**
 * Matrix Multiplication: C = A * B.
 *
 * This sample is a very basic sample that implements matrix multiplication
 * Matrices are stored in row-major order. Matrices are flattened out first
 * 
 * Each block handles one entire row operation: row[0]*col[0]-row[0]*col[4]
 * filling out the first row of the output matrix.
 * Each thread handles a single row operation: row[0] * col[0]
 * 
 * Computed without using shared memory.
 */

#include <cuda_runtime.h>
#include <helper_cuda.h>
//#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
using namespace std;

#define n 5

/**
 * CUDA Kernel Device code
 *
 * Computes the matrix multiplication of A and B into C. The 3 vectors have the same
 * number of elements (n x n).
 * 
 */

__global__ void
matMul(const int  *A, const int *B, int *C,  int N)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockIdx.x;
	int col = threadIdx.x;
	int colSize = blockDim.x;
	int tmpSum = 0;
    if (threadID < N*N)
    {
		// Each thread computes one element of the block sub-matrix
		for (int i = 0; i < N; i++) {
			tmpSum += A[row * colSize + i] * B[i * colSize + col];
		}
	}
	C[threadID] = tmpSum;  
}

/**
 * Host main routine
 */
int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int n = 50000;
    printf("Square Matrix Multiplication of %nx%n elements\n", n);

    // Allocate the host input matrix A & B
    int A[n][n], B[n][n];

    // Set the random seed
	srand(time(NULL));

    // Randomly populate the matrix
	for (int r = 0; r < n; r++)
	{
		for (int c = 0; c < n; c++)
		{
			A[r][c] = rand() % 100;
			B[r][c] = rand() % 100;
		}
	}

    // Print matrix A
	cout << "Matrix A:" << endl;
	for (int r = 0; r < n; r++)
	{
		for (int c = 0; c < n; c++)
		{
			cout << A[r][c] << " , ";
		}
		cout << endl;
	}

    // Print matrix B
	cout << "Matrix B:" << endl;
	for (int r = 0; r < n; r++)
	{
		for (int c = 0; c < n; c++)
		{
			cout << B[r][c] << " , ";
		}
		cout << endl;
	}

	// Copy 2D matrix into 1D array, i.e. flatten
	int* A_1D = new int(n * n);
	for (int r = 0; r < n; r++)
	{
		for (int c = 0; c < n; c++)
		{
			A_1D[r * n + c] = A[r][c];
		}
	}

    // Copy 2D matrix into 1D array, i.e. flatten
	int* B_1D = new int(n * n);
	for (int r = 0; r < n; r++)
	{
		for (int c = 0; c < n; c++)
		{
			B_1D[r * n + c] = B[r][c];
		}
	}

    // Print matrix A_1D
	cout << "Matrix in 1D array A_1D:" << endl;
	for (int r = 0; r < n; r++)
	{
		for (int c = 0; c < n; c++)
		{
			cout << A_1D[r * n + c] << " , ";
		}
		cout << endl;
	}

    // Print matrix B_1D
	cout << "Matrix in 1D array B_1D:" << endl;
	for (int r = 0; r < n; r++)
	{
		for (int c = 0; c < n; c++)
		{
			cout << B_1D[r * n + c] << " , ";
		}
		cout << endl;
	}

    // Print the vector length to be used, and compute its size
    int numElements = n*n;
    size_t size = numElements * sizeof(int);
    printf("[Matrix multiplication of %d elements]\n", numElements);

    // Allocate the host input vector A
    int *h_A = (int *)malloc(size);

    // Allocate the host input vector B
    int *h_B = (int *)malloc(size);

    // Allocate the host output vector C
    int *h_C = (int *)malloc(size);

    // Initialize the host input vectors
    for (int i = 0; i < n*n; ++i)
    {
        h_A[i] = A_1D[i];
        h_B[i] = B_1D[i];
    }

    // Allocate the device input vector A
    int *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    int *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
   int *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = n;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    matMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch matMul kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	// Now do the matrix multiplication on the CPU and print
	cout << "C in CPU" << endl;
	int sum;
	int  cpu_C[n][n];
	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			sum = 0;
			for (int k = 0; k < n; k++) {
					sum += A[row][k] * B[k][col];
			}
			cpu_C[row][col] = sum;
			cout << cpu_C[row][col] <<" , ";
		}
		cout << endl;
	}

	cout << "C in GPU:" << endl;
	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
            cout <<  h_C[row * n + col] << " , ";
		   }
		cout << endl;
	}

    printf("Test PASSED\n");

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