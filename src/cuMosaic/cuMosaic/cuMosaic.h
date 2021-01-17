#pragma once

#include <vector>
#include "pixel.h"

using namespace std;

/*
	Quickly reads a .ppm file from the given path and returns it as
	a vector of vector of pixels to represent the base image.
*/
vector<vector<pixel>> ReadPpm(char* path);