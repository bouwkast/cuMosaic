#pragma once
#include <vector>
#include "Coordinate.h"

using namespace std;
class FastPoisson
{

private:
	int _width;
	int _height;
	float _radius;
	/*
	Validates that the given point is at least radius away from any other point created.
	*/
	bool IsValid(Coordinate point, int width, int height, float radius, float cellSize, std::vector<Coordinate>& points, std::vector<std::vector<int>>& grid);
public:
	
	FastPoisson(int width, int height, float radius);
	/*
	Implementation of Dr Robert Bridson's "Fast Poisson Disk Sampling in Arbitrary Dimensions" to randomly, but uniformly, distribute seeds throughout a rectangular plane.
	*/
	std::vector<Coordinate> GenerateSeedsPoisson();
};

