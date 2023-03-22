//
// Created by samuel on 18/03/2023.
//

#ifndef PHYVR_ASSET_H
#define PHYVR_ASSET_H

#include <string>
#include <android/asset_manager.h>

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

std::string read_asset(AAssetManager *mgr, const std::string &file_name);

img_rgb read_png(AAssetManager *mgr, const std::string &png_file_path);

img_grey to_img_grey(img_rgb image);

#endif //PHYVR_ASSET_H
