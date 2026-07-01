//
// Created by samuel on 01/07/2026.
//

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include <SOIL2.h>

#include <arenai_core_tests/utils/local_file_reader.h>

LocalAssetFileReader::LocalAssetFileReader(const std::filesystem::path &path_to_assets)
    : path_to_assets(path_to_assets) {}

std::string LocalAssetFileReader::read_text(const std::filesystem::path &file_path) {
    const std::ifstream file(path_to_assets / file_path);
    if (!file) throw std::runtime_error("Could not open " + file_path.string());

    std::stringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

ImageChannels LocalAssetFileReader::read_png(const std::filesystem::path &png_file_path) {
    int w = 0, h = 0, source_channels = 0;
    constexpr int out_channels = 4;

    unsigned char *data = SOIL_load_image(
        (path_to_assets / png_file_path).string().c_str(), &w, &h, &source_channels,
        SOIL_LOAD_RGBA);

    if (!data) {
        throw std::runtime_error(
            std::string("SOIL2: impossible de lire l'image: ") + png_file_path.string() + " — "
            + SOIL_last_result());
    }

    ImageChannels out{};
    out.width = w;
    out.height = h;
    out.channels = out_channels;
    out.pixels = std::vector<uint8_t>(w * h * out_channels);

    std::memcpy(out.pixels.data(), data, w * h * out_channels);

    free(data);

    return out;
}
