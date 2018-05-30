//
// Created by samuel on 25/05/18.
//

#ifndef PHYVR_ASSETS_H
#define PHYVR_ASSETS_H

#include <string>
#include <android/asset_manager.h>

std::string getFileText(AAssetManager *mgr, std::string fileName);

#endif //PHYVR_ASSETS_H
