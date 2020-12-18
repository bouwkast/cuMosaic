#pragma once
#include "color.h"
/*
	Representation of a "pixel" within an image rgb color information 
	and whether or not the pixel is a seed.

	Note: there is now row/col properties - this is to save on memory and should
	be stored externally to the pixel.
*/
struct pixel {
	color color;		// 3 bytes
	unsigned char seed; // 1 byte (0 == not seed; 1 == seed)
};