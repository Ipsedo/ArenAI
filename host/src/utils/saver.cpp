//
// Created by samuel on 03/10/2025.
//

#include "./saver.h"

#include <fstream>

#include <nlohmann/json.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

std::string dtype_to_string(const c10::ScalarType dt) {
    switch (dt) {
        case c10::kFloat: return "float32";
        case c10::kDouble: return "float64";
        case c10::kHalf: return "float16";
        case c10::kBFloat16: return "bfloat16";
        case c10::kByte: return "uint8";
        case c10::kChar: return "int8";
        case c10::kShort: return "int16";
        case c10::kInt: return "int32";
        case c10::kLong: return "int64";
        case c10::kBool: return "bool";
        default: throw std::runtime_error("Unsupported dtype");
    }
}

size_t dtype_nbytes(const c10::ScalarType dt) {
    switch (dt) {
        case c10::kFloat: return 4;
        case c10::kDouble: return 8;
        case c10::kHalf: return 2;
        case c10::kBFloat16: return 2;
        case c10::kByte: return 1;
        case c10::kChar: return 1;
        case c10::kShort: return 2;
        case c10::kInt: return 4;
        case c10::kLong: return 8;
        case c10::kBool: return 1;
        default: throw std::runtime_error("Unsupported dtype");
    }
}

void save_tensor_binary(const torch::Tensor &t, const std::filesystem::path &path) {
    const auto contig = t.contiguous();
    TORCH_CHECK(!contig.is_sparse(), "Sparse tensors not supported in this saver");
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open file for writing: " + path.string());
    ofs.write(
        reinterpret_cast<const char *>(contig.data_ptr()),
        contig.numel() * dtype_nbytes(contig.scalar_type()));
}

void export_state_dict_neutral(
    const std::shared_ptr<torch::nn::Module> &m, const std::filesystem::path &outdir) {
    std::filesystem::create_directories(outdir);

    nlohmann::json manifest;
    manifest["tensors"] = nlohmann::json::array();

    auto add_entry = [&](const std::string &name, const torch::Tensor &t, bool is_buffer) {
        const torch::Tensor cpu_t = t.detach().to(torch::kCPU);
        std::vector<int64_t> shape(cpu_t.sizes().begin(), cpu_t.sizes().end());
        std::string fname = name;
        // remplace les '.' par '_' pour un nom de fichier safe
        std::ranges::replace(fname, '.', '_');
        std::filesystem::path binpath = outdir / (fname + ".bin");
        save_tensor_binary(cpu_t, binpath);

        nlohmann::json j;
        j["name"] = name;
        j["file"] = binpath.filename().string();
        j["dtype"] = dtype_to_string(cpu_t.scalar_type());
        j["shape"] = shape;
        j["is_buffer"] = is_buffer;
        manifest["tensors"].push_back(j);
    };

    for (const auto &p: m->named_parameters(/*recurse=*/true)) add_entry(p.key(), p.value(), false);
    for (const auto &b: m->named_buffers(/*recurse=*/true)) add_entry(b.key(), b.value(), true);

    std::ofstream mf(outdir / "manifest.json");
    mf << manifest.dump(2) << std::endl;
}

/*
 * Save PNG
 */

void save_png_rgb(
    const std::vector<std::vector<std::vector<uint8_t>>> &image, const std::string &filename) {
    if (image.empty() || image[0].empty() || image[0][0].empty()) {
        throw std::runtime_error("image vide");
    }

    const int channels = static_cast<int>(image.size());
    const int height = static_cast<int>(image[0].size());
    const int width = static_cast<int>(image[0][0].size());

    if (channels != 3) { throw std::runtime_error("L'image doit avoir 3 canaux (R,G,B)"); }

    std::vector<uint8_t> buffer(width * height * channels);

    // Remplissage interleavé
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < channels; ++c)
                buffer[(y * width + x) * channels + c] = image[c][y][x];

    // Sauvegarde en PNG
    if (!stbi_write_png(
            filename.c_str(), width, height, channels, buffer.data(), width * channels)) {
        throw std::runtime_error("Erreur lors de la sauvegarde PNG");
    }
}
