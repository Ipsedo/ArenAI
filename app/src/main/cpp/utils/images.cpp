//
// Created by samuel on 19/03/2023.
//

#include "images.h"
#include "logging.h"

img_rgb to_img_rgb(libpng_image image) {
    img_rgb res{};

    res.width = (int) image.width;
    res.height = (int) image.height;
    res.pixels = new char[res.width * res.height * 3];

    for (int row = 0; row < image.height; row++) {
        png_bytep currRow = image.row_ptrs[row];
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

img_grey to_img_grey(libpng_image image) {
    img_grey res{};

    res.pixels = new float[image.width * image.height];

    for (int row = 0; row < image.height; row++)
        for (int col = 0; col < image.width; col++)
            res.pixels[row * image.height + col] =
                    float(image.row_ptrs[row][col]) / 255.f;

    res.width = int(image.width);
    res.height = int(image.height);

    return res;
}
