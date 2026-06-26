//
// Created by samuel on 26/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_IMAGE_WRITER_H
#define ARENAI_TRAIN_HOST_IMAGE_WRITER_H

#include <stdexcept>
#include <string>

#include <stb_image_write.h>
#include <torch/torch.h>

void save_tensor_png(const torch::Tensor &image, const std::string &path);

#endif//ARENAI_TRAIN_HOST_IMAGE_WRITER_H
