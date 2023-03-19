//
// Created by samuel on 18/03/2023.
//

#ifndef PHYVR_ASSET_H
#define PHYVR_ASSET_H

#include <string>
#include <android/asset_manager.h>

#include "images.h"

std::string read_asset(AAssetManager *mgr, const std::string &file_name);

libpng_image read_png(AAssetManager *mgr, const std::string &png_file_path);

#endif //PHYVR_ASSET_H
