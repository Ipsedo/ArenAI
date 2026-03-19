//
// Created by samuel on 03/10/2025.
//

#include <cstring>
#include <fstream>
#include <iostream>

#include <soil2/SOIL2.h>

#include <arenai_train/file_reader.h>

DesktopAssetFileReader::DesktopAssetFileReader(const std::filesystem::path &path_to_assets)
    : path_to_assets(path_to_assets) {}

std::string DesktopAssetFileReader::read_text(const std::string &file_name) {
    std::ifstream file(path_to_assets / file_name);
    if (!file) throw std::runtime_error("Could not open " + file_name);

    std::stringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

ImageChannels DesktopAssetFileReader::read_png(const std::string &png_file_path) {
    int w = 0, h = 0, source_channels = 0;
    constexpr int out_channels = 4;

    unsigned char *data = SOIL_load_image(
        (path_to_assets / png_file_path).c_str(), &w, &h, &source_channels, SOIL_LOAD_RGBA);

    if (!data) {
        throw std::runtime_error(
            std::string("SOIL2: impossible de lire l'image: ") + png_file_path + " — "
            + SOIL_last_result());
    }

    ImageChannels out{};
    out.width = w;
    out.height = h;
    out.channels = out_channels;
    out.pixels = std::vector<uint8_t>(w * h * out_channels);

    std::memcpy(out.pixels.data(), data, w * h * out_channels);

    SOIL_free_image_data(data);

    return out;
}
