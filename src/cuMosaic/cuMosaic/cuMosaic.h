#pragma once

#include <vector>
#include "pixel.h"
#include "coordinate.h"

using namespace std;

/*
	Computes Euclidean distance between two points, but forgoes
	the square root for performance.
*/
unsigned int EuclideanDistanceSquared(int x1, int y1, int x2, int y2);

/*
	After Voronoi computation is completed this can iterate over the
	seeds and turn their color to be black. This allows for the
	seeds to b visible in the output image.
*/
void SetSeedsBlack(pixel* seeds, pixel* grid, int numSeeds, int width);

/*
	Initializes the pixels in the grid to match the input image and
	to reset any computations done in previous Voronoi computations.
	i.e. we run this after computing the serial Voronoi to ensure
	that the GPU Voronoi is correctly functioning.
*/
void ResetGridPixels(pixel* grid, vector<vector<pixel>>, int height, int width);


/*
	Computes the maximum, minimum distance between any pixel and its
	closest seed that was generated by Poisson sampling.
	This distance can then be usedfor local seed search if its
	returned value squared is less than the number of seeds.

	Has a computational complexity of O(nm), where n is the number
	of pixels and m is the number of seeds, so recommended to use
	r^2 for the search radius, which based on observation is more
	than enough to properly compute the local search.
*/
int ComputeSearchRadius(std::vector<coordinate>& points, vector<vector<pixel>>& image);

/*
	Implementation of Dr Robert Bridson's
	"Fast Poisson Disk Sampling in Arbitrary Dimensions"
	to randomly, but uniformly, distribute seeds throughout the image.
*/
vector<coordinate> GenerateSeedsPoisson(int width, int height, float radius);

/*
	Helper function for generate_seeds_poisson.
	Validates that the given point is at least radius
	away from any other point created.
*/
bool IsValid(coordinate point, int width, int height, float radius, float cellSize,
	vector<coordinate>& points, vector<vector<int>>& grid);

/*
	For each pixel in the grid, iterate through all seeds
	and take update the pixel's color to be that of its closest
	seed.
*/
void ComputeVoronoiGlobal(pixel* grid, pixel* seeds, int numSeeds, int height, int width);

/*
	For each pixel in the grid, iterate through a box surrounding it
	of size search_radius^2 pixels and if the pixel
	is a seed calculate the distance from it. The current pixel
	takes the color of its closest seed.
*/
void ComputeVoronoiLocal(pixel* grid, int height, int width, int searchRadius);

/*
	For each seed, create a bounding box around it of size distance^2
	and sample the color of the pixels within. Update the seed's color
	to the mean color of the sampled pixels.

	Recommended that distance is less than the radius given to poisson sampling.
*/
color* ComputeMeanColor(pixel* grid, pixel* seeds, int numSeeds,
	int width, int height, int distance);

/*
Creates numSeeds random colors.
*/
color* CreateRandomColors(int numSeeds);


/*
	Quickly reads a .ppm file from the given path and returns it as
	a vector of vector of pixels to represent the base image.
*/
vector<vector<pixel>> ReadPpm(char* path);

/*
	Quickly writes out a .ppm file to path that represents the grid.
*/
void WritePpm(pixel* grid, char* path, int height, int width);