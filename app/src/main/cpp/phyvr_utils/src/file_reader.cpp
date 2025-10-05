//
// Created by samuel on 28/09/2025.
//

#include <phyvr_utils/file_reader.h>

/*
 * Common
 */

img_grey AbstractFileReader::to_img_grey(img_rgb image) {
    img_grey res{};

    res.pixels = new float[image.width * image.height];

    for (int row = 0; row < image.height; row++)
        for (int col = 0; col < image.width; col++)
            // image.pixels -> RGBA -> * 4
            res.pixels[row * image.width + col] =
                (float(image.pixels[4 * row * image.width + col * 4])
                 + float(image.pixels[4 * row * image.width + col * 4 + 1])
                 + float(image.pixels[4 * row * image.width + col * 4 + 2]))
                / 3.f / 255.f;

    res.width = int(image.width);
    res.height = int(image.height);

    return res;
}
