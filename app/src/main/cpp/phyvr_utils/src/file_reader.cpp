//
// Created by samuel on 28/09/2025.
//

#include <phyvr_utils/file_reader.h>

/*
 * Common
 */

ImageGrey AbstractFileReader::to_img_grey(const ImageChannels &image) {
    ImageGrey res{};

    res.pixels = std::vector<float>(image.width * image.height);

    const int out_channels = std::min(3, image.channels);

    for (int row = 0; row < image.height; row++)
        for (int col = 0; col < image.width; col++) {
            float channels_sum = 0.f;
            for (int c = 0; c < out_channels; c++)
                channels_sum += static_cast<float>(
                    image.pixels[image.channels * row * image.width + col * image.channels + c]);

            res.pixels[row * image.width + col] =
                channels_sum / static_cast<float>(out_channels) / 255.f;
        }

    res.width = image.width;
    res.height = image.height;

    return res;
}
