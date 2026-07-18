//
// Created by samuel on 03/10/2025.
//

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

#include <SOIL2.h>

#include <arenai_train/file_reader.h>

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

    DesktopAssetFileReader::DesktopAssetFileReader(const std::filesystem::path &path_to_resources)
        : path_to_resources(path_to_resources) {}

    std::string DesktopAssetFileReader::read_text(const std::filesystem::path &file_path) {
        // binary: read_text also serves binary assets (TTF fonts for the GUI),
        // and Windows text mode corrupts them (CRLF translation, 0x1A as EOF)
        std::ifstream file(path_to_resources / file_path, std::ios::binary);
        if (!file) throw std::runtime_error("Could not open " + file_path.string());

        std::stringstream buffer;
        buffer << file.rdbuf();

        return buffer.str();
    }

    utils::ImageChannels
    DesktopAssetFileReader::read_png(const std::filesystem::path &png_file_path) {
        int w = 0, h = 0, source_channels = 0;
        constexpr int out_channels = 4;

        unsigned char *data = SOIL_load_image(
            (path_to_resources / png_file_path).string().c_str(), &w, &h, &source_channels,
            SOIL_LOAD_RGBA);

        if (!data) {
            throw std::runtime_error(
                std::string("SOIL2: impossible de lire l'image: ") + png_file_path.string() + " — "
                + SOIL_last_result());
        }

        utils::ImageChannels out{};
        out.width = w;
        out.height = h;
        out.channels = out_channels;
        out.pixels = std::vector<uint8_t>(w * h * out_channels);

        std::memcpy(out.pixels.data(), data, w * h * out_channels);

        free(data);

        return out;
    }

}// namespace arenai::train
