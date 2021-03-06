/*CUDA Thread Indexing*/


//1D Grid of 1D Blocks:
__device__
int getGlobalIdx_1D_1D(){
    return (blockIdx.x * blockDim.x) + threadIdx.x;
}


//1D Grid of 2D Blocks:
__device__
int getGlobalIdx_1D_2D(){
    //Total threads in a block: blockDim.x * blockDim.y
    //Row-wise traversal of threads: threadIdx.y * blockDim.x  
    return blockIdx.x * (blockDim.x * blockDim.y)
            + (threadIdx.y * blockDim.x) + threadIdx.x;  
}


//1D Grid of 3D Blocks:
__device__
int getGlobalIdx_1D_3D(){
    return blockIdx.x * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.y * blockDim.x))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
}


//2D Grid of 1D Blocks:
__device__
int getGlobalIdx_2D_1D(){
    //Row-wise traversal of blocks: blockIdx.y * gridDim.x
    //Moving blockId number of blocks at a time: blockId * blockDim.x
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}



//2D Grid of 2D Blocks:
__device__
int getGlobalIdx_2D_2D(){
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * (blockDim.x * blockDim.y)
                    + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}



//2D Grid of 3D Blocks:
__device__
int getGlobalIdx_2D_3D(){{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                    + (threadIdx.z * (blockDim.x * blockDim.y))
                    + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}


//3D Grid of 1D Blocks:
__device__
int getGlobalIdx_3D_1D(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}


//3D Grid of 2D Blocks:
__device__
int getGlobalIdx_3D_2D(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y)
                    + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}


//3D Grid of 3D Blocks:
__device__
int getGlobalIdx_1D_1D(){
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
                    + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                    + (threadIdx.z * (blockDim.x * blockDim.y))
                    + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}
