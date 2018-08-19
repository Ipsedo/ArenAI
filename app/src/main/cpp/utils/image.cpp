//
// Created by samuel on 17/08/18.
//

#include <cmath>
#include "image.h"
#include <android/log.h>

colored_image toRGBImg(libpng_image image) {
	colored_image res;
	int nbOctet = image.bitdepth / 8;
	res.allpixels = vector<color<int>>();
	for (int row = 0; row < image.height; row++) {
		for (int col = 0; col < image.width; col++) {
			int idx = (row * image.width + col) * image.channels;
			int idxOctet = idx * nbOctet;

			int r = 0, g = 0, b = 0;
			for (int i = image.bitdepth - 1, j = 0; i >= 0; i -= 8, j++) {
				r = image.data[idxOctet + j] << (int) pow(2., i) | r;
				g = image.data[idxOctet + j + nbOctet] << (int) pow(2., i) | g;
				b = image.data[idxOctet + j + 2 * nbOctet] << (int) pow(2., i) | b;
			}
			res.allpixels.push_back(color<int>(r, g, b));
		}
	}
	res.maxValue = (int) pow(2., image.bitdepth) - 1;
	res.width = image.width;
	res.height = image.height;
	return res;
}

normalized_image toGrayImg(libpng_image image) {
	normalized_image res;

	int nbOctet = image.bitdepth / 8;

	res.allpixels = vector<float>();

	float maxValue = 0;
	float imgMax = pow(2., image.bitdepth) - 1;

	for (int row = 0; row < image.height; row++) {
		for (int col = 0; col < image.width; col++) {
			int idx = (row * image.width + col) * image.channels;
			int idxOctet = idx * nbOctet;

			unsigned int px = 0;
			for (int i = 0, j = 0; i < image.bitdepth; i += 8, j++) {
					px = image.data[idxOctet + j] << (int) pow(2., i) | px;
			}
			/*for (int i = 0, j = 0; i < image.bitdepth; i += 8, j++) {
				px = image.data[idxOctet + j] << (int) pow(2., i) | px;
			}*/
			float p = float(px) / imgMax;
			maxValue = p > maxValue ? p : maxValue;
			res.allpixels.push_back(p);
		}
	}

	for (int i = 0; i < res.allpixels.size(); i++)
		res.allpixels[i] /= maxValue;

	res.width = image.width;
	res.height = image.height;
	return res;
}
