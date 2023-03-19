//
// Created by samuel on 19/03/2023.
//

#include "images.h"

img_rgb to_img_rgb(libpng_image image) {
    img_rgb res{};

    res.width = (int) image.width;
    res.height = (int) image.height;
    res.pixels = new char[res.width * res.height * 3];

    for (int row = 0; row < image.height; row++) {
        png_bytep currRow = image.rowPtrs[row];
        for (int col = 0; col < image.width; col++) {
            int idx = col * 3;
            int resId = int(row * image.width + col) * 3;
            res.pixels[resId] = (char) currRow[idx];
            res.pixels[resId + 1] = (char) currRow[idx + 1];
            res.pixels[resId + 2] = (char) currRow[idx + 2];
        }
    }

    return res;
}
