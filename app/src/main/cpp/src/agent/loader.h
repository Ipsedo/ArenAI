//
// Created by samuel on 03/10/2025.
//

#ifndef PHYVR_LOADER_H
#define PHYVR_LOADER_H

#include <string>

#include <android/asset_manager.h>
#include <android_native_app_glue.h>
#include <jni.h>

std::string get_cache_dir(android_app *app);

std::string
copy_asset_to_files(AAssetManager *mgr, const std::string &asset_name, const std::string &dst_dir);

#endif// PHYVR_LOADER_H
