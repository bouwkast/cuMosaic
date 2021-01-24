#pragma once
#include <string>
#include <vector>
#include "Pixel.h"
class FileHelper
{
	std::string inputFilePath;
	std::string outputFilePath;

public:
	/*
		Quickly reads a .ppm file from the given path and returns it as
		a vector of vector of pixels to represent the base image.
	*/
	std::vector<std::vector<Pixel>> ReadPpm(std::string path);

	/*
		Quickly writes out a .ppm file to path that represents the grid.
	*/
	void WritePpm(Pixel* grid, std::string path, int height, int width);
};

