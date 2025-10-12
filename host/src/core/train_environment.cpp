//
// Created by samuel on 03/10/2025.
//

#include "./train_environment.h"

#include <phyvr_utils/cache.h>
#include <phyvr_utils/singleton.h>
#include <phyvr_view/errors.h>
#include <phyvr_view/pbuffer_renderer.h>

#include "../utils/linux_file_reader.h"

TrainTankEnvironment::TrainTankEnvironment(
    const int nb_tanks, const std::filesystem::path &android_assets_path)
    : BaseTanksEnvironment(
        std::make_shared<LinuxAndroidAssetFileReader>(android_assets_path), std::nullptr_t(),
        nb_tanks, 1.f / 30.f) {}

void TrainTankEnvironment::on_draw(
    const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {}

void TrainTankEnvironment::on_reset_physics(const std::unique_ptr<PhysicEngine> &engine) {}

void TrainTankEnvironment::on_reset_drawables(
    const std::unique_ptr<PhysicEngine> &engine,
    const std::shared_ptr<AbstractGLContext> &gl_context) {}

void TrainTankEnvironment::reset_singleton() {
    Singleton<Cache<std::shared_ptr<Shape>>>::get_singleton()->clear();
    Singleton<Cache<std::shared_ptr<Shape>>>::reset_singleton();

    Singleton<Cache<btVector3>>::get_singleton()->clear();
    Singleton<Cache<btVector3>>::reset_singleton();

    const auto cache_collision_shape = Singleton<Cache<btCollisionShape *>>::get_singleton();
    //cache_collision_shape->apply_on_items([](auto s) { delete s; });
    cache_collision_shape->clear();
    Singleton<Cache<btCollisionShape *>>::reset_singleton();

    Singleton<Cache<std::shared_ptr<Program>>>::get_singleton()->clear();
    Singleton<Cache<std::shared_ptr<Program>>>::reset_singleton();
}
