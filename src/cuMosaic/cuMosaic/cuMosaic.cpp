
#include <iostream>
#include <chrono>
#include <fstream>

#include "cuMosaic.h"
#include <string>
#include <sstream>

int main()
{
    std::cout << "START" << std::endl;

    // init rand with random seed
    srand(time(NULL));
}

unsigned int EuclideanDistanceSquared(int x1, int y1, int x2, int y2) {
	return (unsigned int)(pow((x2 - x1), 2)) + (pow((y2 - y1), 2));
}

void SetSeedsBlack(pixel* seeds, pixel* grid, int numSeeds, int width)
{
	for (int i = 0; i < numSeeds; i++) {
		int pos = (seeds[i].row * width) + seeds[i].col;
		grid[pos].color.r = 0;
		grid[pos].color.g = 0;
		grid[pos].color.b = 0;
	}
}

pixel* GenerateRandomSeeds(int width, int height, int numSeeds)
{
	int max = width * height;
	// note seeds are encoded as position in a 1D representation of the 2D grid
	pixel* seeds = new pixel[numSeeds];

	for (int i = 0; i < numSeeds; i++) {
		int row = rand() % height;
		int col = rand() % width;
		seeds[i].row = row;
		seeds[i].col = col;
	}

	return seeds;
}


color* CreateRandomColors(int numSeeds)
{
	color* colors = new color[numSeeds];
	for (int i = 0; i < numSeeds; i++) {
		colors[i].r = rand() % 255;
		colors[i].g = rand() % 255;
		colors[i].b = rand() % 255;

	}
	return colors;
}

color* ComputeMeanColor(pixel* grid, pixel* seeds, int numSeeds, int width, int height, int distance) {
	color* colors = new color[numSeeds];

	for (int i = 0; i < numSeeds; i++) {
		int r = 0;
		int g = 0;
		int b = 0;
		int count = 0;
		int startRow = max(0, (int)seeds[i].row - distance);
		int endRow = min(height - 1, (int)seeds[i].row + distance);

		int startCol = max(0, (int)seeds[i].col - distance);;
		int endCol = min(width - 1, (int)seeds[i].col + distance);
		for (int row = startRow; row <= endRow; row++) {
			for (int col = startCol; col <= endCol; col++) {
				count++;
				int pos = (row * width) + col;
				r += grid[pos].color.r;
				g += grid[pos].color.g;
				b += grid[pos].color.b;
			}
		}

		colors[i].r = (unsigned char)ceil((float)r / (float)count);
		colors[i].g = (unsigned char)ceil((float)g / (float)count);
		colors[i].b = (unsigned char)ceil((float)b / (float)count);
	}


	return colors;
}

vector<vector<pixel>> ReadPpm(char* path) {
	cout << "Reading image via: " << path << endl;
	auto startTime = std::chrono::high_resolution_clock::now();

	ifstream f;
	f.open(path, fstream::in | fstream::binary);

	// header for ppm binary format is still in ASCII
	// NOTE: IF YOU CONVERT IMAGE USING GIMP YOU NEED TO REMOVE THEIR COMMENT

	// first line is magical "P6"
	string readLine;
	std::getline(f, readLine);
	if (readLine != "P6") {
		cout << "Doesn't appear that a binary .ppm file is being read. Please verify that the image is a .ppm and is in non-ASCII format\n";
		f.close();
		exit(1);
	}

	// check for GIMP conversion - GIMP inserts a comment with # on 2nd line
	// if it is GIMP we need to handle it a bit differently

	bool isGimp = false;
	std::getline(f, readLine); // 2nd line
	int width = 0;
	int height = 0;
	if (readLine[0] == 35) { // '#'
		isGimp = true;
	}

	// second line is WIDTH HEIGHT
	if (!isGimp) {
		// already read the 2nd line checking for GIMP conversion
		stringstream lineStream;
		lineStream << readLine;

		lineStream >> width;
		lineStream >> height;
	}
	else {
		f >> width;
		f >> height;
	}

	// third line is MAX_VAL
	std::getline(f, readLine);
	if (isGimp) {
		// this is the WIDTH HEIGHT (already got it)
		std::getline(f, readLine);
	}
	if (readLine != "255") {
		cout << "Expected MAX_VAL of 255, but was " << readLine << endl;
		f.close();
		exit(1);
	}

	vector<vector<pixel>> grid;

	// allocate  space to read the image
	char* imageBuffer = new char[width * height * 3]; // *3 for each color value
	f.read(imageBuffer, width * height * 3);

	for (int row = 0; row < height; row++) {
		grid.push_back(vector<pixel>(width));
		for (int col = 0; col < width; col++) {
			int pos = ((row * width) + col) * 3;
			grid[row][col].row = row;
			grid[row][col].col = col;
			grid[row][col].color.r = imageBuffer[pos];
			grid[row][col].color.g = imageBuffer[(pos + 1)];
			grid[row][col].color.b = imageBuffer[(pos + 2)];
		}
	}

	delete[] imageBuffer;

	f.close();

	auto endTime = std::chrono::high_resolution_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() * 0.001;
	std::cout << "Read input file in " << elapsed << " ms" << std::endl;

	return grid;
}

void WritePpm(pixel* grid, char* path, int height, int width)
{
	cout << "Printing output image to " << path << endl;
	auto startTime = std::chrono::high_resolution_clock::now();
	ofstream f;
	f.open(path, fstream::out | fstream::binary);

	f << "P6" << endl;
	f << width << " " << height << endl;
	f << "255" << endl;

	// prepare the buffer
	char* buffer = new char[height * width * 3];

	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			int bufferPos = ((row * width) + col) * 3;
			int imagePos = (row * width) + col;
			buffer[bufferPos] = grid[imagePos].color.r;
			buffer[bufferPos + 1] = grid[imagePos].color.g;
			buffer[bufferPos + 2] = grid[imagePos].color.b;
		}
	}
	f.write(buffer, height * width * 3);

	f.close();
	delete[] buffer;
	auto endTime = std::chrono::high_resolution_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() * 0.001;
	std::cout << "Printed output image in " << elapsed << " ms" << std::endl;
}