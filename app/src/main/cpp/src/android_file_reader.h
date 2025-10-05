//
// Created by samuel on 29/09/2025.
//

#ifndef PHYVR_ANDROID_FILE_READER_H
#define PHYVR_ANDROID_FILE_READER_H

#include <android/asset_manager.h>

#include <phyvr_utils/file_reader.h>

class AndroidFileReader : public AbstractFileReader {
public:
    explicit AndroidFileReader(AAssetManager *mgr);

    std::string read_text(const std::string &file_name) override;

    img_rgb read_png(const std::string &png_file_path) override;

private:
    AAssetManager *mgr;
};

#endif// PHYVR_ANDROID_FILE_READER_H
