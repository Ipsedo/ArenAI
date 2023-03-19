//
// Created by samuel on 19/03/2023.
//

#ifndef PHYVR_IMAGES_H
#define PHYVR_IMAGES_H

#include <vector>
#include <png.h>

struct img_rgb {
    int width;
    int height;
    char *pixels;
};

struct img_grey {
    int width;
    int height;
    float *pixels;
};

struct libpng_image {
    png_uint_32 width;
    png_uint_32 height;
    png_uint_32 bitdepth;
    png_uint_32 channels;
    char *data;
    png_bytep *row_ptrs;
};

img_rgb to_img_rgb(libpng_image image);

img_grey to_img_grey(libpng_image image);

#endif //PHYVR_IMAGES_H
