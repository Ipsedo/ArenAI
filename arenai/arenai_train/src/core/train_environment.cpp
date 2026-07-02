//
// Created by samuel on 03/10/2025.
//

#include "./train_environment.h"

#include <arenai_train/file_reader.h>
#include <arenai_utils/cache.h>
#include <arenai_utils/singleton.h>
#include <arenai_view/pbuffer_renderer.h>

#include "../metrics/mean_metric.h"
#include "../metrics/std_metric.h"

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

    TrainTankEnvironment::TrainTankEnvironment(
        const std::shared_ptr<view::AbstractGLContext> &gl_context, const int nb_tanks,
        const std::filesystem::path &android_assets_path, const float wanted_frequency,
        const int max_episode_steps, const int vision_height, const int vision_width,
        const int vision_num_threads)
        : core::BaseTanksEnvironment(
            std::make_shared<DesktopAssetFileReader>(android_assets_path), gl_context, nb_tanks,
            wanted_frequency, vision_height, vision_width, vision_num_threads, false),
          wanted_frequency(wanted_frequency),
          max_frames_without_hit(static_cast<int>(30.f / wanted_frequency)),
          remaining_frames(nb_tanks, max_frames_without_hit),
          nb_frames_added_when_hit(static_cast<int>(5.f / wanted_frequency)), nb_tanks(nb_tanks),
          nb_steps(0), done(nb_tanks, false), already_done(nb_tanks, false),
          max_episode_steps(max_episode_steps),
          episode_step_mean_nb_metric(std::make_shared<MeanMetric>("s_μ", 32, 1)),
          episode_step_std_nb_metric(std::make_shared<StdMetric>("s_σ", 32)) {}

    std::vector<std::tuple<core::State, core::Reward, core::IsDone>>
    TrainTankEnvironment::step(const float time_delta, const std::vector<core::Action> &actions) {

        auto step_result = core::BaseTanksEnvironment::step(time_delta, actions);

        const auto has_hit = apply_on_factories<std::vector<bool>>([&](const auto &factories) {
            std::vector<bool> has_hit_result;
            has_hit_result.reserve(nb_tanks);
            for (const auto &factory: factories)
                has_hit_result.push_back(factory->has_hit_other_tank());
            return has_hit_result;
        });

        already_done = done;

        // natural ending (timeout, death)
        for (int i = 0; i < step_result.size(); i++) {
            remaining_frames[i]--;
            if (has_hit[i]) remaining_frames[i] += nb_frames_added_when_hit;

            if (const auto &[state, reward, is_done] = step_result[i];
                remaining_frames[i] <= 0 || is_done) {
                const float timeout_penalty = remaining_frames[i] <= 0 ? 0.5f : 0.f;
                step_result[i] = {state, reward - timeout_penalty, true};
                done[i] = true;
            }
        }

        // detect winner
        for (int i = 0; i < step_result.size(); i++) {
            if (done[i]) continue;

            if (const long nb_not_done = std::ranges::count(done, false); nb_not_done == 1) {
                const auto &[state, reward, is_done] = step_result[i];
                if (only_one_tank_alive())
                    step_result[i] = {state, reward + 2.f, true}; // winner réel
                else step_result[i] = {state, reward + 1.f, true};// timeout winner
                done[i] = true;
            }
        }

        nb_steps++;

        return step_result;
    }

    void TrainTankEnvironment::on_draw(
        const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {}

    void TrainTankEnvironment::on_reset_physics(
        const std::unique_ptr<model::AbstractPhysicEngine> &engine) {
        remaining_frames = std::vector(nb_tanks, max_frames_without_hit);

        const float nb_seconds = static_cast<float>(nb_steps) * wanted_frequency;
        episode_step_mean_nb_metric->add(nb_seconds);
        episode_step_std_nb_metric->add(nb_seconds);

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

    bool TrainTankEnvironment::are_all_done() {
        return std::accumulate(
            done.begin(), done.end(), true,
            [](const int acc, const bool curr_done) { return acc && curr_done; });
    }

    bool TrainTankEnvironment::is_tank_factory_already_done(const int tank_factory_index) {
        return already_done[tank_factory_index];
    }

    bool TrainTankEnvironment::is_episode_terminated() {
        return are_all_done() || nb_steps > max_episode_steps;
    }

    void TrainTankEnvironment::on_reset_drawables(
        const std::unique_ptr<model::AbstractPhysicEngine> &engine,
        const std::shared_ptr<view::AbstractGLContext> &gl_context) {}

    void TrainTankEnvironment::reset_singleton() {
        utils::Singleton<utils::Cache<std::shared_ptr<model::Shape>>>::get_singleton()->clear();
        utils::Singleton<utils::Cache<std::shared_ptr<model::Shape>>>::reset_singleton();

        utils::Singleton<utils::Cache<std::shared_ptr<view::Program>>>::get_singleton()->clear();
        utils::Singleton<utils::Cache<std::shared_ptr<view::Program>>>::reset_singleton();
    }

    std::vector<std::shared_ptr<AbstractMetric>> TrainTankEnvironment::get_metrics() const {
        return {episode_step_mean_nb_metric, episode_step_std_nb_metric};
    }

}// namespace arenai::train
