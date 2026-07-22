//
// Created by samuel on 26/06/2026.
//

// stb_image_write is header-only: this translation unit provides the
// implementation behind the declarations image_writer.h pulls in. STATIC
// keeps the symbols internal — soil2 already exports its own stb copy.
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "./image_writer.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    void save_tensor_png(const torch::Tensor &image, const std::string &path) {
        if (image.dim() != 3 || image.size(0) != 3)
            throw std::invalid_argument("Tensor attendu de shape (3, H, W)");

        const int64_t height = image.size(1);
        const int64_t width = image.size(2);

        torch::Tensor img = image.detach().to(torch::kCPU);

        img = img.to(torch::kUInt8);

        img = img.permute({1, 2, 0}).contiguous();

        if (const int stride_bytes = static_cast<int>(width * 3);
            stbi_write_png(
                path.c_str(), static_cast<int>(width), static_cast<int>(height),
                3,// composantes RGB
                img.data_ptr<uint8_t>(), stride_bytes)
            == 0) {
            throw std::runtime_error("Échec de l'écriture PNG : " + path);
        }
    }

}// namespace arenai::agent
