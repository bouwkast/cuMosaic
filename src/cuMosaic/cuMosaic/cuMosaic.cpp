#define _USE_MATH_DEFINES
#include <iostream>
#include <chrono>
#include <fstream>

#include "cuMosaic.h"
#include <string>
#include <sstream>
#include <random>

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

vector<coordinate> GenerateSeedsPoisson(int width, int height, float radius) {
	auto startTime = std::chrono::high_resolution_clock::now();
	// increasing k == more samples
	int k = 30; // rejection limit (recommended from the paper)
	// cell size to guarantee that each cell can have at most 1 sample
	float cellSize = radius / sqrtf(2);

	// these are the dimensions of the accelerator grid
	int correctedWidth = (int)(ceil((float)width / cellSize));
	int correctedHeight = (int)(ceil((float)height / cellSize));

	// grid to speed up rejection of generated samples
	vector<vector<int>> backingGrid;
	for (int row = 0; row < correctedHeight; row++) {
		backingGrid.push_back(vector<int>(correctedWidth));
		//for (int col = 0; col < correctedWidth; col++) {
		//	backingGrid[row].push_back(0); // indicates empty
		//}
	}

	vector<coordinate> points;
	vector<coordinate> activePoints;

	// for uniform float dist: https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
	std::random_device randomDevice;
	std::mt19937 gen(randomDevice());
	std::uniform_real<float> uniformFloat(0.0, 1.0);
	std::uniform_real<float> uniformRadialDistance(radius, 2 * radius);

	// init by adding an active point at the center
	coordinate initializerPoint;
	initializerPoint.x = width / 2;
	initializerPoint.y = height / 2;
	activePoints.push_back(initializerPoint);

	// while we have active points generate additional points
	while (!activePoints.empty()) {
		// choose one of the active points to generate points around
		int index = rand() % activePoints.size();
		bool accept = false;
		coordinate origin = activePoints[index]; // where we will generate points around

		for (int i = 0; i < k; i++) {
			float angle = uniformFloat(gen) * M_PI * 2;
			coordinate direction;
			direction.x = sin(angle);
			direction.y = cos(angle);

			coordinate generatedPoint;
			// get a distance between radius and 2*radius
			float distance = uniformRadialDistance(gen); // distance both x and y
			direction.x = direction.x * distance;
			direction.y = direction.y * distance;
			generatedPoint.x = origin.x + direction.x;
			generatedPoint.y = origin.y + direction.y;

			if (IsValid(generatedPoint, width, height, radius, cellSize, points, backingGrid)) {
				points.push_back(generatedPoint);
				activePoints.push_back(generatedPoint);
				int backingX = (int)(generatedPoint.x / cellSize);
				int backingY = (int)(generatedPoint.y / cellSize);
				backingGrid[backingY][backingX] = points.size();
				accept = true;
				break;
			}

		}
		if (!accept) {
			activePoints.erase(activePoints.begin() + index); // was too close to another point
		}
	}

	auto endTime = std::chrono::high_resolution_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() * 0.001;
	std::cout << "Generated Poisson points in " << elapsed << " ms" << std::endl;

	return points;
}

bool IsValid(coordinate point, int width, int height, float radius, float cellSize, vector<coordinate>& points, vector<vector<int>>& grid) {
	// x and y needs to be within our boundaries
	if (point.x >= 0 && point.x < width && point.y >= 0 && point.y < height) {
		int backingX = (int)(point.x / cellSize);
		int backingY = (int)(point.y / cellSize);

		// search around this point with a distance of radius
		int startX = max(0, backingX - 2);
		int endX = min(backingX + 2, (int)grid[0].size() - 1);
		int startY = max(0, backingY - 2);
		int endY = min(backingY + 2, (int)grid.size() - 1);

		for (int x = startX; x <= endX; x++) {
			for (int y = startY; y <= endY; y++) {
				int pointIndex = grid[y][x] - 1;

				if (pointIndex != -1) {
					// means that we are taken
					// compute distance
					float distance = EuclideanDistanceSquared(point.x, point.y, points[pointIndex].x, points[pointIndex].y);
					if (distance < radius * radius) {
						return false;
					}
				}
			}
		}

		return true;
	}

	return false;
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