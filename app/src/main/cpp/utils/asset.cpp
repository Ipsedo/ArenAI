//
// Created by samuel on 18/03/2023.
//

#include "asset.h"

#include <android/asset_manager.h>

std::string read_asset(AAssetManager *mgr, const std::string &file_name) {
    // Open your file
    AAsset *file = AAssetManager_open(mgr, file_name.c_str(), AASSET_MODE_BUFFER);
    // Get the file length
    off_t fileLength = AAsset_getLength(file);

    // Allocate memory to read your file
    char *fileContent = new char[fileLength + 1];

    // Read your file
    AAsset_read(file, fileContent, size_t(fileLength));
    // For safety you can add a 0 terminating character at the end of your file ...
    fileContent[fileLength] = '\0';

    // Do whatever you want with the content of the file
    AAsset_close(file);

    std::string res = std::string(fileContent);

    delete[] fileContent;

    return res;
}
