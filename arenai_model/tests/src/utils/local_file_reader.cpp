//
// Created by samuel on 01/07/2026.
//

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <arenai_model_tests/utils/local_file_reader.h>

using namespace arenai;
using namespace arenai::utils;

LocalAssetFileReader::LocalAssetFileReader(const std::filesystem::path &path_to_assets)
    : path_to_assets(path_to_assets) {}

std::string LocalAssetFileReader::read_text(const std::filesystem::path &file_path) {
    // binary: matches DesktopAssetFileReader (Windows text mode corrupts binary assets)
    const std::ifstream file(path_to_assets / file_path, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open " + file_path.string());

    std::stringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}

ImageChannels LocalAssetFileReader::read_png(const std::filesystem::path &png_file_path) {
    throw std::runtime_error("read_png not implemented in model tests");
}
