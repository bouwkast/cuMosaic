#pragma once
#include "Color.h"
/*
	Representation of a "pixel" within an image rgb color information
	and whether or not the pixel is a seed.

	Note: the row/col values need to be stored somewhere else as this doubles the size size of this from 4 bytes to 8 bytes,
	but it makes some calculations much easier.
*/
struct Pixel {
	unsigned short row; // 2 bytes
	unsigned short col; // 2 bytes
	Color color;		// 3 bytes
	unsigned char seed; // 1 byte (0 == not seed; 1 == seed)
};