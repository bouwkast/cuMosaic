#include <math.h>

unsigned int EuclideanDistanceSquared(int x1, int y1, int x2, int y2) {
	return (unsigned int)(pow((x2 - x1), 2)) + (pow((y2 - y1), 2));
}