//
// Created by samuel on 03/10/2025.
//

#include "./linux_file_reader.h"

#include <fstream>
#include <iostream>

#include <soil2/SOIL2.h>

LinuxAndroidAssetFileReader::LinuxAndroidAssetFileReader(
    const std::filesystem::path &path_to_assets)
    : path_to_assets(path_to_assets) {}

std::string LinuxAndroidAssetFileReader::read_text(const std::string &file_name) {
    std::ifstream file(path_to_assets / file_name);
    if (!file) throw std::runtime_error("Could not open " + file_name);

    std::stringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

ImageChannels LinuxAndroidAssetFileReader::read_png(const std::string &png_file_path) {
    int w = 0, h = 0, channels = 0;

    unsigned char *data = SOIL_load_image(
        (path_to_assets / png_file_path).c_str(), &w, &h, &channels, SOIL_LOAD_RGBA);

    if (!data) {
        throw std::runtime_error(
            std::string("SOIL2: impossible de lire l'image: ") + png_file_path + " — "
            + SOIL_last_result());
    }

    ImageChannels out{};
    out.width = w;
    out.height = h;
    out.channels = channels;
    out.pixels = reinterpret_cast<char *>(data);

    return out;
}
