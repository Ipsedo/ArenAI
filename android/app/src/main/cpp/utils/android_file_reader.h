//
// Created by samuel on 29/09/2025.
//

#ifndef ARENAI_ANDROID_FILE_READER_H
#define ARENAI_ANDROID_FILE_READER_H

#include <android/asset_manager.h>

#include <arenai_utils/file_reader.h>

class AndroidFileReader : public AbstractFileReader {
public:
    explicit AndroidFileReader(AAssetManager *mgr);

    std::string read_text(const std::string &file_name) override;

    ImageChannels read_png(const std::string &png_file_path) override;

private:
    AAssetManager *mgr;
};

#endif// ARENAI_ANDROID_FILE_READER_H
