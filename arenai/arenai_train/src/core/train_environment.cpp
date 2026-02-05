//
// Created by samuel on 03/10/2025.
//

#include "./train_environment.h"

#include <algorithm>

#include <arenai_utils/cache.h>
#include <arenai_utils/singleton.h>
#include <arenai_view/errors.h>
#include <arenai_view/pbuffer_renderer.h>

#include "../utils/linux_file_reader.h"

TrainTankEnvironment::TrainTankEnvironment(
    const int nb_tanks, const std::filesystem::path &android_assets_path,
    const float wanted_frequency)
    : BaseTanksEnvironment(
        std::make_shared<LinuxAndroidAssetFileReader>(android_assets_path), std::nullptr_t(),
        nb_tanks, wanted_frequency, false),
      max_frames_without_shoot(static_cast<int>(90.f / wanted_frequency)),
      remaining_frames(nb_tanks, max_frames_without_shoot),
      nb_frames_added_when_shoot(static_cast<int>(5.f / wanted_frequency)), nb_tanks(nb_tanks) {}

std::vector<std::tuple<State, Reward, IsDone>> TrainTankEnvironment::step(
    const float time_delta, std::future<std::vector<Action>> &actions_future) {

    auto step_result = BaseTanksEnvironment::step(time_delta, actions_future);

    const auto has_shoot = apply_on_factories<std::vector<bool>>([&](const auto &factories) {
        std::vector<bool> has_shoot_result;
        has_shoot_result.reserve(nb_tanks);
        for (const auto &factory: factories)
            has_shoot_result.push_back(factory->has_shoot_other_tank());
        return has_shoot_result;
    });

    for (int i = 0; i < step_result.size(); i++) {
        remaining_frames[i]--;

        const auto &[state, reward, __] = step_result[i];

        if (has_shoot[i]) remaining_frames[i] += nb_frames_added_when_shoot;

        if (remaining_frames[i] <= 0) step_result[i] = {state, reward, true};
    }

    return step_result;
}

std::vector<Reward> TrainTankEnvironment::get_potential_rewards() {
    return apply_on_factories<std::vector<Reward>>([&](const auto &factories) {
        std::vector<Reward> potential_rewards;
        potential_rewards.reserve(factories.size());

        for (const auto &factory: factories)
            potential_rewards.push_back(factory->get_potential_reward(factories));

        return potential_rewards;
    });
}

void TrainTankEnvironment::on_draw(
    const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {}

void TrainTankEnvironment::on_reset_physics(const std::unique_ptr<PhysicEngine> &engine) {
    remaining_frames = std::vector(nb_tanks, max_frames_without_shoot);
}

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
