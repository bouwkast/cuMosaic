#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "pixel.h"
#include <stdio.h>
#include <limits>
#include <chrono>
#include <stdlib.h> 
#include <iostream>

#define BLOCK_SIZE 1024

__device__ int EuclideanDistanceSquared(int x1, int y1, int x2, int y2) {
	return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
}

__global__ void VoronoiGlobalSearch(pixel* cudaGrid, pixel* cudaSeeds, int gridHeight, int gridWidth, int numSeeds) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos < (gridHeight * gridWidth)) {
		if (cudaGrid[pos].seed) {
			return;
		}
		unsigned int minDistance = INT_MAX;
		pixel closestSeed;
		for (int seedPos = 0; seedPos < numSeeds; seedPos++) {
			int distance = EuclideanDistanceSquared(cudaGrid[pos].row, cudaGrid[pos].col, cudaSeeds[seedPos].row, cudaSeeds[seedPos].col);

			if (distance <= minDistance) {
				minDistance = distance;
				closestSeed = cudaSeeds[seedPos];
			}
		}

		// set grid position closest seed
		cudaGrid[pos].color = closestSeed.color;
	}
}

__global__ void VoronoiLocalSearch(pixel* cudaGrid, pixel* cudaSeeds, int gridHeight, int gridWidth, int searchRadius, int numSeeds) {
	// compute grid position
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos >= (gridHeight * gridWidth)) {
		return; // can't access this memory
	}

	if (cudaGrid[pos].seed) {
		return;
	}

	// compute boundaries for search box
	int startRow = (int)cudaGrid[pos].row - searchRadius;
	if (startRow < 0) {
		startRow = 0;
	}

	int endRow = (int)cudaGrid[pos].row + searchRadius;
	if (endRow >= gridHeight) {
		endRow = gridHeight - 1;
	}

	int startCol = (int)cudaGrid[pos].col - searchRadius;
	if (startCol < 0) {
		startCol = 0;
	}

	int endCol = (int)cudaGrid[pos].col + searchRadius;
	if (endCol >= gridWidth) {
		endCol = gridWidth - 1;
	}

	unsigned int minDistance = INT_MAX;
	bool success = false;
	pixel closestSeed;

	// iterate through local search space and find closest seed
	for (int boxRow = startRow; boxRow <= endRow; boxRow++) {
		for (int boxCol = startCol; boxCol <= endCol; boxCol++) {
			int boxPos = (boxRow * gridWidth) + boxCol;
			if (cudaGrid[boxPos].seed) {
				int dist = EuclideanDistanceSquared(cudaGrid[pos].row, cudaGrid[pos].col, cudaGrid[boxPos].row, cudaGrid[boxPos].col);

				if (dist <= minDistance) {
					minDistance = dist;
					success = true;
					closestSeed = cudaGrid[boxPos];
				}
			}
		}
	}

	if (success) {
		cudaGrid[pos].color = closestSeed.color;
		return;
	}

	// local search failed - fallback to global strategy (same as the other search)
	// Note - never actually seen the local search fail, but it is a possibility
	minDistance = INT_MAX;
	for (int seedPos = 0; seedPos < numSeeds; seedPos++) {
		int distance = EuclideanDistanceSquared(cudaGrid[pos].row, cudaGrid[pos].col, cudaSeeds[seedPos].row, cudaSeeds[seedPos].col);

		if (distance <= minDistance) {
			minDistance = distance;
			closestSeed = cudaSeeds[seedPos];
		}
	}

	// set grid position closest seed
	cudaGrid[pos].color = closestSeed.color;
}

extern "C" void CudaComputeVoronoi(pixel * grid, pixel * seeds, int gridHeight, int gridWidth, int numSeeds, int searchRadius) {
	cudaError_t result;

	// cuda related data
	pixel* cudaGrid;
	pixel* cudaSeeds;

	int gridSize = gridWidth * gridHeight;

	// select our 0 GPU to run on
	result = cudaSetDevice(0);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(1);
	}

	// allocate data for our grid, seeds, and colors
	result = cudaMalloc((void**)&cudaGrid, sizeof(pixel) * gridSize);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for grid allocation.");
		exit(1);
	}

	result = cudaMalloc((void**)&cudaSeeds, sizeof(pixel) * numSeeds);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed for seed allocation.");
		exit(1);
	}

	// copy over our grid
	result = cudaMemcpy(cudaGrid, grid, sizeof(pixel) * gridSize, cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed while downloading grid data to device.");
		exit(1);
	}

	// copy over our seeds
	result = cudaMemcpy(cudaSeeds, seeds, sizeof(pixel) * numSeeds, cudaMemcpyHostToDevice);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed while downloading seed data to device.");
		exit(1);
	}
	// set execution configuration on GPU

	dim3 dimblock(BLOCK_SIZE);
	dim3 dimgrid(ceil((float)gridSize / BLOCK_SIZE));

	// compute voronoi
	if (searchRadius > 0) {
		VoronoiLocalSearch << <dimgrid, dimblock >> > (cudaGrid, cudaSeeds, gridHeight, gridWidth, searchRadius, numSeeds);

	}
	else {
		VoronoiGlobalSearch << <dimgrid, dimblock >> > (cudaGrid, cudaSeeds, gridHeight, gridWidth, numSeeds);
	}


	// check to see if there were any errors
	result = cudaGetLastError();
	if (result != cudaSuccess) {
		fprintf(stderr, "voronoi launch failed: %s\n", cudaGetErrorString(result));
		exit(1);
	}
	else {
		std::cout << "kernel has been launched" << std::endl;
	}

	// copy over our grid data
	result = cudaMemcpy(grid, cudaGrid, sizeof(pixel) * gridSize, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed while uploading grid data from the device.");
		exit(1);
	}
	else {
		std::cout << "finished copying data over" << std::endl;
	}

	// release grid memory allocation
	result = cudaFree(cudaGrid);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaFree failed while freeing cuda_grid!");
		exit(1);
	}

	// release seed memory allocation
	result = cudaFree(cudaSeeds);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaFree failed while freeing cuda_grid!");
		exit(1);
	}
}