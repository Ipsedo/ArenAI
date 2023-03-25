//
// Created by samuel on 25/05/18.
//

#ifndef PHYVR_ASSETS_H
#define PHYVR_ASSETS_H

#include "image.h"
#include <android/asset_manager.h>
#include <string>

std::string getFileText(AAssetManager *mgr, std::string fileName);

libpng_image readPNG(AAssetManager *mgr, std::string pngName);

#endif // PHYVR_ASSETS_H
