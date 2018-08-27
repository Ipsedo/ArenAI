//
// Created by samuel on 17/08/18.
//

#include <cmath>
#include "image.h"
#include <android/log.h>

colored_image toColoredImg(libpng_image image) {
	colored_image res;
	res.allpixels = vector<color<int>>();

	for (int row = 0; row < image.height; row++) {
		png_bytep currRow = image.rowPtrs[row];
		for (int col = 0; col < image.width; col++) {
			int idx = col * 3;
			res.allpixels.push_back(color<int>(currRow[idx], currRow[idx + 1], currRow[idx + 2]));
		}
	}

	res.width = image.width;
	res.height = image.height;
	return res;
}

normalized_image toGrayImg(libpng_image image) {
	normalized_image res;

	res.allpixels = vector<float>();

	float max = 0.f;

	for (int row = 0; row < image.height; row++) {
		png_bytep currRow = image.rowPtrs[row];
		for (int col = 0; col < image.width; col++) {
			int px = currRow[col];
			max = max < px ? px : max;
			res.allpixels.push_back(px);
		}
	}
	for (int i = 0; i < res.allpixels.size(); i++) {
		res.allpixels[i] /= max;
	}

	res.width = image.width;
	res.height = image.height;
	return res;
}


imgRGB toImgRGB(libpng_image image) {
	imgRGB res;

	res.width = image.width;
	res.height = image.height;
	res.pixels = new char[res.width * res.height * 3];

	for (int row = 0; row < image.height; row++) {
		png_bytep currRow = image.rowPtrs[row];
		for (int col = 0; col < image.width; col++) {
			int idx = col * 3;
			int resId = (row * image.width + col) * 3;
			res.pixels[resId] = currRow[idx];
			res.pixels[resId + 1] = currRow[idx + 1];
			res.pixels[resId + 2] = currRow[idx + 2];
		}
	}
	return res;
}
