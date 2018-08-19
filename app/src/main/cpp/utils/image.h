//
// Created by samuel on 17/08/18.
//

#ifndef PHYVR_IMAGE_H
#define PHYVR_IMAGE_H

#include <png.h>
#include <vector>

using namespace std;

struct normalized_image {
	int width;
	int height;
	vector<float> allpixels;//
};

template<typename T> class color {
public:
	T r;
	T g;
	T b;
	color(T r, T g, T b) : r(r), g(g), b(b) {};
};

struct colored_image {
	int width;
	int height;
	int maxValue;
	vector<color<int>> allpixels;
};

struct libpng_image {
	png_uint_32 width;
	png_uint_32 height;
	png_uint_32 bitdepth;
	png_uint_32 channels;
	char* data;
	png_bytep* rowPtrs;
};

colored_image toRGBImg(libpng_image image);

normalized_image toGrayImg(libpng_image image);

#endif //PHYVR_IMAGE_H
