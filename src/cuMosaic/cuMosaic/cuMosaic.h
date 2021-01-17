#pragma once

#include <vector>
#include "pixel.h"

using namespace std;

/*
	Quickly reads a .ppm file from the given path and returns it as
	a vector of vector of pixels to represent the base image.
*/
vector<vector<pixel>> ReadPpm(char* path);

/*
	Quickly writes out a .ppm file to path that represents the grid.
*/
void WritePpm(pixel* grid, char* path, int height, int width);