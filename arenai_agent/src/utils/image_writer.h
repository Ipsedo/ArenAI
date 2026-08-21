//
// Created by samuel on 26/06/2026.
//

#ifndef ARENAI_AGENT_HOST_IMAGE_WRITER_H
#define ARENAI_AGENT_HOST_IMAGE_WRITER_H

#include <string>

#include <torch/torch.h>

namespace arenai::agent {

    void save_tensor_png(const torch::Tensor &image, const std::string &path);

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_IMAGE_WRITER_H
