#define _USE_MATH_DEFINES
#include <iostream>
#include <chrono>
#include <fstream>

#include "cuMosaic.h"
#include <string>
#include <sstream>
#include <random>


// if true we assume local search will be 2r
constexpr auto FAST_CALC = true;
extern "C" void CudaComputeVoronoi(pixel * grid, pixel * seeds, int gridHeight, int gridWidth, int numSeeds, int searchRadius);

static void ShowUsage(std::string programName) {
	std::cerr << "Usage: " << programName << " <option(s)>\n"
		<< "Options(-short,--long):\n"
		<< "\t-h,--help\t\tDisplays this usage message\n"
		<< "\t-i,--input\t\tThe path to the input image encoded in binary PPM format. e.g. \"imgs\\test.ppm\"\n"
		<< "\t-o,--output\t\tThe path to store the output image in binary PPM format. e.g. \"imgs\\output.ppm\"\n"
		<< "\t-d,--distance\t\tThe minimum allowable distance (roughly in pixels) between any seed generated.\n\t\t\t\t\t\tFloat within range (0, X*Y-1] where X is the image width and Y is the image height.\n\t\t\t\t\t\tSmaller values & larger images will take longer to process.";
	// todo implement this http://www.cplusplus.com/articles/DEN36Up4/
}

static std::string ExtractArgumentValue(int index, std::string argument, int argc, char* argv[]) {
	if (index + 1 >= argc) {
		std::cerr << argument << " requires a value, please refer to the --help output for further information." << std::endl;
		ShowUsage(argv[0]);
		throw 1; // TODO is this standard practice in C++?
	}

	std::string extractedValue = argv[index + 1];
	return extractedValue;
}

int main(int argc, char *argv[])
{
	// we gonna need the following arguments "input.ppm" "output.ppm"  (min pixels between each seed)10

	float radius = 10.0;
	std::string inputFile = "input.ppm";
	std::string outputFile = "output.ppm";

	if (argc != 7) {
		ShowUsage(argv[0]);
		// TODO make this follow the standard "USAGE" format
		return 1;
	}

	/// TODO this is a basic implementation, none of the inputs are really validated here, which isn't really an issue
	// extract the command options, this allows them to be in any order
	for (int i = 1; i < argc; ++i) {
		std::string argument = argv[i];
		std::string value;
		if ((argument == "-h") || (argument == "--help")) {
			ShowUsage(argv[0]);
			return 0;
		}
		else if ((argument == "-i") || (argument == "--input")) {
			value = ExtractArgumentValue(i++, argument, argc, argv);
			inputFile = value;
		}
		else if ((argument == "-o") || argument == "--output") {
			value = ExtractArgumentValue(i++, argument, argc, argv);
			outputFile = value;
		}
		else if ((argument == "-d") || argument == "--distance") {
			value = ExtractArgumentValue(i++, argument, argc, argv);
			// TODO I don't fully understand why I need to do the "size_type" here, should look into that
			std::string::size_type sz;     // alias of size_t
			radius = std::stof(value, &sz); // minimum distance between any two points
		}
		else {
			std::cerr << "Unknown option entered: " << argument << std::endl;
			ShowUsage(argv[0]);
			return 1;
		}
	}

	std::cout << "START" << std::endl;

	// START Initialization of image/poisson/mean color sampling

	// init rand with random seed
	srand(time(NULL));

	// hardcoding image paths for testing
	vector<vector<pixel>> image = ReadPpm(inputFile);

	// extract out the image dimensions
	int height = image.size();
	int width = image[0].size();

	cout << "Image read with height x width of " << height << " x " << width << " pixels." << endl;

	cout << "Generating seeds with Fast Poisson Disk Sampling with a radius of " << radius << endl;

	vector<coordinate> points = GenerateSeedsPoisson(width, height, radius);
	int numSeeds = points.size();
	cout << "Poisson sampling generated " << numSeeds << " seeds." << endl;

	// compute the maximal minimum distance between any two seeds
	int searchRadius = 0;
	if (FAST_CALC) {
		/*
			We are assuming here that our Poisson sampling algorithm was able to reliably place points
			within 2r of any other point. Meaning that the maximal minimum distance between any two points
			is less than 2r.
			While the algorithm doesn't guarantee this, through observation it is quite true and
			most of the points it generates are typically within roughly 1.1r of eachother.

		*/
		searchRadius = (int)ceil(radius * 2);
	}
	else {
		searchRadius = ComputeSearchRadius(points, image); // O(nm) 
	}

	cout << "Modified search radius for computing voronoi cells is " << searchRadius << endl;

	bool useLocalSearch = (int)searchRadius * (int)searchRadius < points.size();
	if (useLocalSearch) {
		cout << "Using local seed search." << endl;
	}
	else {
		searchRadius = -1;
		cout << "Global searching through all seeds." << endl;
	}
	pixel* grid = new pixel[width * height];

	// init grid
	ResetGridPixels(grid, image, height, width);

	// init seeds from poisson points
	pixel* seeds = new pixel[numSeeds];
	for (int i = 0; i < numSeeds; i++) {
		// important to take the floor; otherwise we could be out of bounds
		seeds[i].row = (int)floor(points[i].y);
		seeds[i].col = (int)floor(points[i].x);
	}

	// initialize the colors
	int colorSamplingRadius = (int)(ceil(radius / 2));
	color* colors = ComputeMeanColor(grid, seeds, numSeeds, width, height, colorSamplingRadius);
	//color* colors = CreateRandomColors(numSeeds);

	// update seeds with their colors
	for (int i = 0; i < numSeeds; i++) {
		seeds[i].color = colors[i];
	}

	// iterate through seeds, mark grid positions that are to be used as seeds
	for (int i = 0; i < numSeeds; i++) {
		int pos = (seeds[i].row * width) + seeds[i].col;
		grid[pos].seed = 1;
		grid[pos].color = seeds[i].color;
	}

	// END Initialization

	cout << "Starting serial test" << endl;
	// compute Voronoi
	auto startTime = std::chrono::high_resolution_clock::now();
	/*if (useLocalSearch) {
		ComputeVoronoiLocal(grid, height, width, (int)searchRadius);
	}
	else {
		ComputeVoronoiGlobal(grid, seeds, numSeeds, height, width);
	}*/
	auto endTime = std::chrono::high_resolution_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() * 0.001;
	std::cout << "Computed Voronoi in " << elapsed << " ms" << std::endl;

	// setting seeds black to validate that seeds were properly placed
	SetSeedsBlack(seeds, grid, numSeeds, width);

	// print image
	WritePpm(grid, outputFile, height, width);

	ResetGridPixels(grid, image, height, width);
	// make sure our seeds are still correct
	for (int i = 0; i < numSeeds; i++) {
		int pos = (seeds[i].row * width) + seeds[i].col;
		grid[pos].seed = 1;
		grid[pos].color = seeds[i].color;
	}

	cout << "Starting GPU test" << endl;
	startTime = std::chrono::high_resolution_clock::now();
	CudaComputeVoronoi(grid, seeds, height, width, numSeeds, searchRadius);
	endTime = std::chrono::high_resolution_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() * 0.001;
	std::cout << "Computed Voronoi GPU in " << elapsed << " ms" << std::endl;

	// setting seeds black to validate that seeds were properly placed
	//SetSeedsBlack(seeds, grid, numSeeds, width);

	// print image
	WritePpm(grid, outputFile, height, width);

	std::cout << "END" << std::endl;

	delete[] grid;
	delete[] seeds;
	delete[] colors;
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

void ResetGridPixels(pixel* grid, vector<vector<pixel>> image, int height, int width)
{
	for (unsigned short row = 0; row < height; row++) {
		for (unsigned short col = 0; col < width; col++) {
			int pos = (row * width) + col;
			grid[pos].row = row;
			grid[pos].col = col;
			grid[pos].color.r = image[row][col].color.r;
			grid[pos].color.g = image[row][col].color.g;
			grid[pos].color.b = image[row][col].color.b;
			grid[pos].seed = 0;
		}
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

int ComputeSearchRadius(std::vector<coordinate>& points, vector<vector<pixel>>& image)
{
	auto startTime = std::chrono::high_resolution_clock::now();
	float searchRadius = 0.0;
	for (int row = 0; row < image.size(); row++) {
		for (int col = 0; col < image[0].size(); col++) {
			unsigned int minDistance = INT_MAX;
			for (int i = 0; i < points.size(); i++) {
				int seedRow = (int)floor(points[i].x);
				int seedCol = (int)floor(points[i].y);
				if (row == seedRow && col == seedCol) {
					continue; // skip as this pixel is a seed
				}

				unsigned int distance = EuclideanDistanceSquared(seedRow, seedCol, row, col);
				if (distance < minDistance) {
					minDistance = distance;
				}
			}
			// computed the distance to every seed from this pixel
			if (minDistance > searchRadius) {
				searchRadius = minDistance;
			}
		}
	}

	// need to square root the value to get the correct distance
	searchRadius = sqrtf(searchRadius);

	auto endTime = std::chrono::high_resolution_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() * 0.001;
	std::cout << "Computed search radius in " << elapsed << " ms" << std::endl;

	cout << "Maximum distance between a pixel and its nearest seed: " << searchRadius << endl;
	searchRadius = ceil(searchRadius) + 1; // eg  10.1 -> 12 (guaranteed we will be searching slightly more than we need to)
	return (int)searchRadius;
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

vector<vector<pixel>> ReadPpm(std::string path) {
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

void WritePpm(pixel* grid, std::string path , int height, int width)
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

void ComputeVoronoiLocal(pixel* grid, int height, int width, int searchRadius) {
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			int pos = (row * width) + col;
			float minDistance = FLT_MAX;
			pixel closestSeed;
			// create the bounds for the bounding box around the current pixel
			int startRow = max(0, (int)grid[pos].row - searchRadius);
			int endRow = min(height - 1, (int)grid[pos].row + searchRadius);

			int startCol = max(0, (int)grid[pos].col - searchRadius);
			int endCol = min(width - 1, (int)grid[pos].col + searchRadius);

			for (int boxRow = startRow; boxRow <= endRow; boxRow++) {
				for (int boxCol = startCol; boxCol <= endCol; boxCol++) {
					int boxPos = (boxRow * width) + boxCol;
					if (grid[boxPos].seed) {
						// compute distance
						unsigned int distance = EuclideanDistanceSquared(row, col, boxRow, boxCol);
						if (distance <= minDistance) {
							minDistance = distance;
							closestSeed = grid[boxPos];
						}
					} // else it isn't so we continue on
				}
			}
			grid[pos].color = closestSeed.color;
		}
	}
}

void ComputeVoronoiGlobal(pixel* grid, pixel* seeds, int numSeeds, int height, int width)
{
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			int pos = (row * width) + col;
			float minDistance = FLT_MAX;
			pixel closestSeed;
			for (int seedPos = 0; seedPos < numSeeds; seedPos++) {
				unsigned int distance = EuclideanDistanceSquared(row, col, seeds[seedPos].row, seeds[seedPos].col);
				if (distance <= minDistance) {
					minDistance = distance;
					closestSeed = seeds[seedPos];
				}
			}
			grid[pos].color = closestSeed.color;
		}
	}
}