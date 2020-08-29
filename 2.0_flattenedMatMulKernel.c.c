/**
 * Matrix multiplication CUDA Kernel with flattened matrices
*/

threadRow = blockIdx.y * blockDim.y + threadIdx.y;
threadCol = blockIdx.x * blockDim.x + threadIdx.x;

if ((N * ROW + COL) < N * N){
    int sum = 0;
    
    for (int k = 0; k < N; k++){
        sum += A[threadRow * N + k] * B[k * N + threadCol];
    }
C[N * threadRow + threadCol] = sum;