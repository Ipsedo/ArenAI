//
// Created by samuel on 03/10/2025.
//

#include "./train_environment.h"

#include <phyvr_view/framebuffer_renderer.h>

#include "./utils/linux_file_reader.h"

TrainTankEnvironment::TrainTankEnvironment(const int nb_tanks, const int threads_num)
    : BaseTanksEnvironment(
      std::make_shared<LinuxAndroidAssetFileReader>(std::filesystem::path("")),
      std::make_shared<PBufferGLContext>(), nb_tanks, threads_num) {}

void TrainTankEnvironment::on_draw(
  const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {}

void TrainTankEnvironment::on_reset_physics(const std::shared_ptr<PhysicEngine> &engine) {}

void TrainTankEnvironment::on_reset_drawables(
  const std::shared_ptr<PhysicEngine> &engine,
  const std::shared_ptr<AbstractGLContext> &gl_context) {}
