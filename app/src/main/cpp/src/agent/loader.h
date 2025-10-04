//
// Created by samuel on 03/10/2025.
//

#ifndef PHYVR_LOADER_H
#define PHYVR_LOADER_H

#include <android/asset_manager.h>
#include <string>

std::string copy_asset_to_files(AAssetManager *mgr, const char *asset_name,
                                const char *dst_dir);

#endif // PHYVR_LOADER_H
