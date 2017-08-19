/**
* Copyright 2016 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <cudnn.h>
#include <cuda.h>
#include <stdio.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
extern "C" void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cudnnErrCheck(stat) { cudnnErrCheck_((stat), __FILE__, __LINE__); }
extern "C" void cudnnErrCheck_(cudnnStatus_t stat, const char *file, int line) {
   if (stat != CUDNN_STATUS_SUCCESS) {
      fprintf(stderr, "cuDNN Error: %s %s %d\n", cudnnGetErrorString(stat), file, line);
   }
}

__global__ void initGPUData_ker(float *data, int numElements, float value) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid < numElements) {
      data[tid] = value;
   }
}

extern "C" void initGPUData(float *data, int numElements, float value) {
   dim3 gridDim;
   dim3 blockDim;
   
   blockDim.x = 1024;
   gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;
   
   initGPUData_ker <<< gridDim, blockDim >>> (data, numElements, value);
}

extern "C" int cudamain() { return 0; }