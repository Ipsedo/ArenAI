//
// Created by samuel on 11/03/2026.
//

#include "./game_environment.h"

#include <arenai_train/file_reader.h>

#include "../view/glfw_gl_context.h"

DesktopGameEnvironment::DesktopGameEnvironment(
    const std::string &assets_folder_path, const int window_width, const int window_height,
    const int nb_tanks)
    : BaseTanksEnvironment(
        std::make_shared<DesktopAssetFileReader>(assets_folder_path),
        std::make_shared<GlfwGlContext>(window_width, window_height), nb_tanks, 1.f / 30.f, true),
      player_renderer(std::make_unique<PlayerRenderer>(
          gl_context, window_width, window_height, glm::vec3(0.f), std::nullptr_t())) {}
