//
// Created by samuel on 03/10/2025.
//

#include "./train_environment.h"

#include <arenai_train/file_reader.h>
#include <arenai_utils/cache.h>
#include <arenai_utils/singleton.h>
#include <arenai_view/pbuffer_renderer.h>

TrainTankEnvironment::TrainTankEnvironment(
    const std::shared_ptr<AbstractGLContext> &gl_context, const int nb_tanks,
    const std::filesystem::path &android_assets_path, const float wanted_frequency,
    const int max_episode_steps)
    : BaseTanksEnvironment(
        std::make_shared<DesktopAssetFileReader>(android_assets_path), gl_context, nb_tanks,
        wanted_frequency, false),
      wanted_frequency(wanted_frequency),
      max_frames_without_hit(static_cast<int>(30.f / wanted_frequency)),
      remaining_frames(nb_tanks, max_frames_without_hit),
      nb_frames_added_when_hit(static_cast<int>(5.f / wanted_frequency)), nb_tanks(nb_tanks),
      nb_steps(0), done(nb_tanks, false), already_done(nb_tanks, false),
      max_episode_steps(max_episode_steps),
      episode_step_nb_metric(std::make_shared<Metric>("seconds", 32, 1)) {}

std::vector<std::tuple<State, Reward, IsDone>>
TrainTankEnvironment::step(const float time_delta, const std::vector<Action> &actions) {

    auto step_result = BaseTanksEnvironment::step(time_delta, actions);

    const auto has_hit = apply_on_factories<std::vector<bool>>([&](const auto &factories) {
        std::vector<bool> has_hit_result;
        has_hit_result.reserve(nb_tanks);
        for (const auto &factory: factories)
            has_hit_result.push_back(factory->has_hit_other_tank());
        return has_hit_result;
    });

    for (int i = 0; i < step_result.size(); i++) {
        if (done[i] && !already_done[i]) already_done[i] = true;

        remaining_frames[i]--;
        if (has_hit[i]) remaining_frames[i] += nb_frames_added_when_hit;

        const auto &[state, reward, is_done] = step_result[i];

        // timeout or done
        if (remaining_frames[i] <= 0 || is_done) {
            const float timeout_penalty = remaining_frames[i] <= 0 ? 0.5f : 0.f;

            step_result[i] = {state, reward - timeout_penalty, true};

            done[i] = true;
        }

        if (!done[i] && only_one_tank_not_already_done()) {
            // timeout winner
            step_result[i] = {state, reward + 1.f, true};
            done[i] = true;
        }

        if (!done[i] && only_one_tank_alive()) {
            // winner
            step_result[i] = {state, reward + 2.f, true};
            done[i] = true;
        }
    }

    nb_steps++;

    return step_result;
}

std::vector<float> TrainTankEnvironment::get_phi_vector() {
    return apply_on_factories<std::vector<float>>([](const auto &factories) {
        std::vector<float> phi_vector;
        for (const auto &factory: factories) { phi_vector.push_back(factory->get_phi(factories)); }
        return phi_vector;
    });
}

void TrainTankEnvironment::on_draw(
    const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {}

void TrainTankEnvironment::on_reset_physics(const std::unique_ptr<PhysicEngine> &engine) {
    remaining_frames = std::vector(nb_tanks, max_frames_without_hit);

    episode_step_nb_metric->add(static_cast<float>(nb_steps) * wanted_frequency);
    nb_steps = 0;

    already_done = std::vector(nb_tanks, false);
    done = std::vector(nb_tanks, false);
}

bool TrainTankEnvironment::only_one_tank_alive() {
    return apply_on_factories<bool>([](const auto &factories) {
        int nb_alive = 0;
        for (const auto &factory: factories) nb_alive += static_cast<int>(!factory->is_dead());

        return nb_alive == 1;
    });
}

bool TrainTankEnvironment::only_one_tank_not_already_done() {
    return std::accumulate(
               already_done.begin(), already_done.end(), 0,
               [](const int acc, const bool done) { return acc + static_cast<int>(!done); })
           == 1;
}

bool TrainTankEnvironment::are_all_done() {
    return std::accumulate(done.begin(), done.end(), true, [](const int acc, const bool curr_done) {
        return acc && curr_done;
    });
}

bool TrainTankEnvironment::is_tank_factory_already_done(const int tank_factory_index) {
    return already_done[tank_factory_index];
}

bool TrainTankEnvironment::is_episode_terminated() {
    return are_all_done() || nb_steps > max_episode_steps;
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

std::vector<std::shared_ptr<Metric>> TrainTankEnvironment::get_metrics() const {
    return {episode_step_nb_metric};
}
