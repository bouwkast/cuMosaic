#define _USE_MATH_DEFINES
#include "FastPoisson.h"
#include "DistanceFunctions.h"
#include <iostream>
#include <chrono>
#include <random>
using namespace std;

bool FastPoisson::IsValid(Coordinate point, int width, int height, float radius, float cellSize, std::vector<Coordinate>& points, std::vector<std::vector<int>>& grid)
{
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

FastPoisson::FastPoisson(int width, int height, float radius)
{
    _width = width;
    _height = height;
    _radius = radius;
}

std::vector<Coordinate> FastPoisson::GenerateSeedsPoisson()
{
	auto startTime = std::chrono::high_resolution_clock::now();
	// increasing k == more samples
	int k = 30; // rejection limit (recommended from the paper)
	// cell size to guarantee that each cell can have at most 1 sample
	float cellSize = _radius / sqrtf(2);

	// these are the dimensions of the accelerator grid
	int correctedWidth = (int)(ceil((float)_width / cellSize));
	int correctedHeight = (int)(ceil((float)_height / cellSize));

	// grid to speed up rejection of generated samples
	vector<vector<int>> backingGrid;
	for (int row = 0; row < correctedHeight; row++) {
		backingGrid.push_back(vector<int>(correctedWidth));
		//for (int col = 0; col < correctedWidth; col++) {
		//	backingGrid[row].push_back(0); // indicates empty
		//}
	}

	vector<Coordinate> points;
	vector<Coordinate> activePoints;

	// for uniform float dist: https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
	std::random_device randomDevice;
	std::mt19937 gen(randomDevice());
	std::uniform_real<float> uniformFloat(0.0, 1.0);
	std::uniform_real<float> uniformRadialDistance(_radius, 2 * _radius);

	// init by adding an active point at the center
	Coordinate initializerPoint;
	initializerPoint.x = _width / 2;
	initializerPoint.y = _height / 2;
	activePoints.push_back(initializerPoint);

	// while we have active points generate additional points
	while (!activePoints.empty()) {
		// choose one of the active points to generate points around
		int index = rand() % activePoints.size();
		bool accept = false;
		Coordinate origin = activePoints[index]; // where we will generate points around

		for (int i = 0; i < k; i++) {
			float angle = uniformFloat(gen) * M_PI * 2;
			Coordinate direction;
			direction.x = sin(angle);
			direction.y = cos(angle);

			Coordinate generatedPoint;
			// get a distance between radius and 2*radius
			float distance = uniformRadialDistance(gen); // distance both x and y
			direction.x = direction.x * distance;
			direction.y = direction.y * distance;
			generatedPoint.x = origin.x + direction.x;
			generatedPoint.y = origin.y + direction.y;

			if (IsValid(generatedPoint, _width, _height, _radius, cellSize, points, backingGrid)) {
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
