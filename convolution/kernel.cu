#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>

const int TILE_SIZE = 1024;
const int MAX_MASK_WIDTH = 10;
__constant__ float const_m[MAX_MASK_WIDTH];

cudaError_t convWithCuda(float *P, const float *N, const float *M, const int Mask_Width, const int Width);

__global__ void convKernel_1(float *P, float *N, float *M, int Mask_Width, int Width) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	float Pvalue = 0;
	int N_start_point = i - (Mask_Width / 2);
	for (int j = 0; j < Mask_Width; j++) {
		if (N_start_point + j >= 0 && N_start_point + j < Width) {
			Pvalue += N[N_start_point + j] * M[j];
		}
	}
	P[i] = Pvalue;
}

__global__ void convKernel_2(float *P, float *N, int Mask_Width, int Width) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	float Pvalue = 0;
	int N_start_point = i - (Mask_Width / 2);
	for (int j = 0; j < Mask_Width; j++) {
		if (N_start_point + j >= 0 && N_start_point + j < Width) {
			Pvalue += N[N_start_point + j] * const_m[j];
		}
	}
	P[i] = Pvalue;
}

__global__ void convKernel_3(float *P, float *N, int Mask_Width, int Width) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];
	int n = Mask_Width / 2;


	//filling left side of N_ds
	int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
	if (threadIdx.x >= blockDim.x - n) {
		N_ds[threadIdx.x - (blockDim.x - n)] =
			(halo_index_left < 0) ? 0 : N[halo_index_left];
	}
	//filling main part of N_ds
	N_ds[n + threadIdx.x] = N[blockIdx.x*blockDim.x + threadIdx.x];
	//filling right side of N_ds
	int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
	if (threadIdx.x < n) {
		N_ds[n + blockDim.x + threadIdx.x] =
			(halo_index_right >= Width) ? 0 : N[halo_index_right];
	}
	__syncthreads();


	//calculating p ...
	float Pvalue = 0;
	for (int j = 0; j < Mask_Width; j++) {
		Pvalue += N_ds[threadIdx.x + j] * const_m[j];
	}
	P[i] = Pvalue;
}


int main()
{
	const int Width = 10;// 0000000;
	const int Mask_Width = 10;
	float *N = (float *)malloc(Width * sizeof(float));
	float *M = (float *)malloc(Mask_Width * sizeof(float));
	float *P = (float *)malloc(Width * sizeof(float));

	for (int i = 0; i < Width; i++) {
		N[i] = 1;
	}
	for (int i = 0; i < Mask_Width; i++) {
		M[i] = 0.1;
	}

	// Add vectors in parallel.
	cudaError_t cudaStatus = convWithCuda(P, N, M, Mask_Width, Width);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "convWithCuda failed!\n");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	system("pause");
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t convWithCuda(float *P, const float *N, const float *M, const int Mask_Width, const int Width)
{
	float *dev_n = 0;
	float *dev_m = 0;
	float *dev_p = 0;
	cudaError_t cudaStatus;
	cudaEvent_t start;
	cudaEventCreate(&start);
	cudaEvent_t stop;
	cudaEventCreate(&stop);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_p, Width * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_n, Width * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_m, Width * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	/*
	cudaStatus = cudaMalloc((void**)&const_m, Mask_Width * sizeof(float));
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed!");
	goto Error;
	}
	

	cudaMemcpyToSymbol(const_m, M, Mask_Width * sizeof(int));*/

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_n, N, Width * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	
	cudaStatus = cudaMemcpy(dev_m, M, Mask_Width * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
	}
	

	// Launch a kernel on the GPU with one thread for each element.
	int gridSize = Width / TILE_SIZE + 1;
	dim3 grid(gridSize, 1, 1);
	dim3 thread(TILE_SIZE, 1, 1);

	printf("%d\n", gridSize);

	cudaEventRecord(start, NULL);

	convKernel_1 <<< grid, thread >>>(dev_p, dev_n, dev_m, Mask_Width, Width);

	cudaEventRecord(stop, NULL);

	cudaStatus = cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
	cudaStatus = cudaEventElapsedTime(&msecTotal, start, stop);
	printf("%f", msecTotal);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaStatus), cudaStatus, __LINE__);
		exit(EXIT_FAILURE);
	}


	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(P, dev_p, Width * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_p);
	cudaFree(dev_n);
	cudaFree(const_m);

	return cudaStatus;
}
