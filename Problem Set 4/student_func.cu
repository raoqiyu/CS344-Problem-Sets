//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>


/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#define numBits 2
#define numBins (1<<numBits)


__global__
void histogram(const unsigned int * const d_in, unsigned int * const d_out, const int nthBit, const size_t numElems){

    //use shared memory to load the whole block data
    extern __shared__ unsigned int tempElems[];

    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;


    if(idx < numElems){
        tempElems[tid] = d_in[idx];
        __syncthreads();
        unsigned int nthBin = (tempElems[tid]>>nthBit)&(numBins-1);
        atomicAdd(&d_out[nthBin],1);
    }
}
//----------------------------------------------------------------------------------
// Hillis Steele Scan
__global__ 
void scan(const unsigned int * const d_in, unsigned int *d_out, unsigned int *d_blockLastElems, const size_t numElems)
{
    //use shared memory to load the whole block data
    extern __shared__ unsigned int temp[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x*blockDim.x + tid;

    if(idx >= numElems)
        return;
    temp[tid] = d_in[idx];
    __syncthreads();

    for(unsigned int stride = 1; stride < blockDim.x; stride <<= 1){
        unsigned int temp_val = temp[tid];
        __syncthreads();

        if(tid + stride < blockDim.x)
            temp[tid+stride] += temp_val;
        __syncthreads();
    }

    // exclusive scan  
    d_out[idx] = tid > 0 ? temp[tid-1] : 0;

    if(tid == (blockDim.x-1))
        d_blockLastElems[blockIdx.x] = temp[tid];

}



__global__
void add(const unsigned int * const d_in, unsigned int * const d_out, const size_t numElems){
    unsigned int bIdx = blockIdx.x;
    unsigned int idx = bIdx*blockDim.x + threadIdx.x;

    if(idx < numElems){
        d_out[idx] += d_in[bIdx];
    }
}
// Exclusive Prefix Sum of Histogram(support a large array, not just in one block)
// 1) first do scan on each block 
// 2) then do scan one all block's last elements
// 3) then add block's last element to its next block
void prefixSum(unsigned int *d_in, unsigned int *d_out, const size_t numElems) 
{
    const dim3 blockSize(min(1024, (int)numElems));
    const dim3 gridSize(ceil((float)numElems/blockSize.x));


    unsigned int *d_blockLastElems;
    checkCudaErrors(cudaMalloc((void**)&d_blockLastElems, gridSize.x*sizeof(unsigned int)));

    // 1) first do scan on each block 
    scan<<<gridSize, blockSize, blockSize.x*sizeof(unsigned int)>>>(d_in, d_out, d_blockLastElems, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    if(gridSize.x > 1){
        // 2) then do scan one all block's last elements
        prefixSum(d_blockLastElems, d_blockLastElems, gridSize.x);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // 3) then add block's last element to its next block
        add<<<gridSize, blockSize>>>(d_blockLastElems, d_out, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }
    checkCudaErrors(cudaFree(d_blockLastElems));
}
//----------------------------------------------------------------------------------
__global__
void map(const unsigned int * const d_in, unsigned int * const d_out, const size_t numElems, const int mask, const int nthBit)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < numElems){
        d_out[idx] = ((d_in[idx]>>nthBit)&(numBins-1)) == mask;
    }
}
__global__
void movebyBins(unsigned int* const d_inputVals,unsigned int* const d_inputPos,
          unsigned int* const d_outputVals,unsigned int* const d_outputPos,
          unsigned int* const d_binElems, unsigned int *d_binScan , unsigned int *d_binHistogram, const size_t numElems, const int mask)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < numElems && d_binElems[idx] == 1){
        unsigned int outputIdx = d_binHistogram[mask] + d_binScan[idx];
        d_outputVals[outputIdx] = d_inputVals[idx];
        d_outputPos[outputIdx] = d_inputPos[idx];
    }
}

void your_sort(unsigned int*  d_inputVals,
               unsigned int*  d_inputPos,
               unsigned int*  d_outputVals,
               unsigned int*  d_outputPos,
               const size_t numElems)
{ 

    const dim3 blockSize(1024);
    const dim3 gridSize(ceil((float)numElems/1024));

    unsigned int *d_binHistogram, *d_binScan, *d_binElems;
    checkCudaErrors(cudaMalloc((void**)&d_binHistogram, numBins*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&d_binScan, numElems*sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&d_binElems, numElems*sizeof(unsigned int)));

    for(int i = 0; i < 8 * (int)sizeof(unsigned int); i += numBits){
        checkCudaErrors(cudaMemset(d_binHistogram,0, numBins*sizeof(unsigned int)));

        // 1) Histogram of the number of occurrences of the i-th bit
        histogram<<<gridSize, blockSize, blockSize.x*sizeof(unsigned int)>>>(d_inputVals, d_binHistogram, i, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // 2) Exclusive Prefix Sum of Histogram
        prefixSum(d_binHistogram, d_binHistogram, numBins);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

        // 3) Determine relative offset of each digit
        // 4) Combine the results of steps 2 & 3 to determine the final output location for each element and move it there
        for(int j = 0; j < numBins; j++){
            map<<<gridSize, blockSize>>>(d_inputVals, d_binElems, numElems, j, i);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

            prefixSum(d_binElems, d_binScan, numElems);
            movebyBins<<<gridSize,blockSize>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, d_binElems,d_binScan,
                                                d_binHistogram, numElems, j);
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        }
        std::swap(d_inputPos, d_outputPos);
        std::swap(d_inputVals, d_outputVals);

    }
    cudaMemcpy(d_outputVals, d_inputVals, numElems*sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_outputPos, d_inputPos, numElems*sizeof(int), cudaMemcpyDeviceToDevice);
    checkCudaErrors(cudaFree(d_binHistogram));
    checkCudaErrors(cudaFree(d_binScan)); 
    checkCudaErrors(cudaFree(d_binElems)); 


}
