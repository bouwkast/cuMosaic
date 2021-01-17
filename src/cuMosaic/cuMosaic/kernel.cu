#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "pixel.h"

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

