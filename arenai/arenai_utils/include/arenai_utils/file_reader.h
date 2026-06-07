//
// Created by samuel on 28/09/2025.
//

#ifndef ARENAI_FILE_READER_H
#define ARENAI_FILE_READER_H

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

struct ImageChannels {
    int width;
    int height;
    int channels;
    std::vector<uint8_t> pixels;
};

struct ImageGrey {
    int width;
    int height;
    std::vector<float> pixels;
};

class AbstractFileReader {
public:
    virtual ~AbstractFileReader() = default;

    virtual std::string read_text(const std::filesystem::path &file_path) = 0;
    virtual ImageChannels read_png(const std::filesystem::path &png_file_path) = 0;

    static ImageGrey to_img_grey(const ImageChannels &image);
};

#endif// ARENAI_FILE_READER_H
