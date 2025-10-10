//
// Created by samuel on 03/10/2025.
//

#include "./train_environment.h"

#include <phyvr_view/framebuffer_renderer.h>

#include "./utils/linux_file_reader.h"

TrainTankEnvironment::TrainTankEnvironment(
    const int nb_tanks, EGLDisplay display, const std::filesystem::path &android_assets_path)
    : BaseTanksEnvironment(
        std::make_shared<LinuxAndroidAssetFileReader>(android_assets_path),
        std::make_shared<PBufferGLContext>(display), nb_tanks, 1.f / 30.f) {}

void TrainTankEnvironment::on_draw(
    const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {}

void TrainTankEnvironment::on_reset_physics(const std::unique_ptr<PhysicEngine> &engine) {}

void TrainTankEnvironment::on_reset_drawables(
    const std::unique_ptr<PhysicEngine> &engine,
    const std::shared_ptr<AbstractGLContext> &gl_context) {}
