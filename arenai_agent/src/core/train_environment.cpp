//
// Created by samuel on 03/10/2025.
//

#include "./train_environment.h"

#include <algorithm>
#include <numeric>

#include <arenai_agent/file_reader.h>
#include <arenai_utils/cache.h>
#include <arenai_utils/singleton.h>
#include <arenai_view/backend.h>

#include "../metrics/mean_metric.h"
#include "../metrics/std_metric.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    TrainTankEnvironment::TrainTankEnvironment(
        const std::shared_ptr<view::AbstractGraphicBackend> &graphics_backend, const int nb_tanks,
        const std::filesystem::path &android_assets_path, const float wanted_frequency,
        const int max_episode_steps, const int vision_height, const int vision_width,
        const int vision_num_threads)
        : BaseTanksEnvironment(
            std::make_shared<DesktopAssetFileReader>(android_assets_path), graphics_backend,
            nb_tanks, wanted_frequency, vision_height, vision_width, vision_num_threads, false),
          wanted_frequency(wanted_frequency),
          max_frames_without_hit(static_cast<int>(30.f / wanted_frequency)),
          remaining_frames(nb_tanks, max_frames_without_hit),
          nb_frames_added_when_hit(static_cast<int>(1.f / wanted_frequency)),
          nb_frames_added_when_kill(static_cast<int>(5.f / wanted_frequency)), nb_tanks(nb_tanks),
          nb_steps(0), done(nb_tanks, false), already_done(nb_tanks, false),
          max_episode_steps(max_episode_steps),
          reward_metric(
              std::make_shared<MeanMetric>("r", 4 * nb_tanks * max_episode_steps, 3, true)),
          episode_step_mean_nb_metric(std::make_shared<MeanMetric>("s_μ", 32, 1)),
          episode_step_std_nb_metric(std::make_shared<StdMetric>("s_σ", 32)),
          fire_metric(std::make_shared<MeanMetric>("fire", 256, 2)),
          hit_metric(std::make_shared<MeanMetric>("hit", 256, 2, true)),
          kill_metric(std::make_shared<MeanMetric>("kill", 16, 1)),
          hits_per_kill_metric(std::make_shared<MeanMetric>("h/k", 16, 1)), nb_kills_episode(0),
          nb_hits_episode(0) {}

    std::vector<std::tuple<core::State, core::Reward, core::IsDone>>
    TrainTankEnvironment::step(const float time_delta, const std::vector<core::Action> &actions) {

        // tanks flagged done on a previous step already emitted their terminal transition:
        // mark them so the caller can skip their post-mortem steps
        already_done = done;

        auto step_result = BaseTanksEnvironment::step(time_delta, actions);

        const auto has_hit = apply_on_factories<std::vector<bool>>([&](const auto &factories) {
            std::vector<bool> has_hit_result;
            has_hit_result.reserve(nb_tanks);
            for (const auto &factory: factories)
                has_hit_result.push_back(factory->consume_has_hit());
            return has_hit_result;
        });

        const auto has_kill = apply_on_factories<std::vector<bool>>([&](const auto &factories) {
            std::vector<bool> has_kill_result;
            has_kill_result.reserve(nb_tanks);
            for (const auto &factory: factories)
                has_kill_result.push_back(factory->consume_has_kill());
            return has_kill_result;
        });

        const auto has_fired = apply_on_factories<std::vector<bool>>([&](const auto &factories) {
            std::vector<bool> has_fired_result;
            has_fired_result.reserve(nb_tanks);
            for (const auto &factory: factories)
                has_fired_result.push_back(factory->consume_has_fire());
            return has_fired_result;
        });

        const auto is_suicide = apply_on_factories<std::vector<bool>>([&](const auto &factories) {
            std::vector<bool> is_suicide_result;
            is_suicide_result.reserve(nb_tanks);
            for (const auto &factory: factories) is_suicide_result.push_back(factory->is_suicide());
            return is_suicide_result;
        });

        // fire / hit frequencies (per second, per tank that acted this step)
        int nb_acting = 0, nb_fires = 0, nb_hits = 0;
        for (int i = 0; i < nb_tanks; i++) {
            if (already_done[i]) continue;
            nb_acting++;
            nb_fires += has_fired[i] ? 1 : 0;
            nb_hits += has_hit[i] ? 1 : 0;
        }
        nb_hits_episode += nb_hits;

        if (nb_acting > 0) {
            fire_metric->add(
                static_cast<float>(nb_fires) / (static_cast<float>(nb_acting) * wanted_frequency));
            hit_metric->add(
                static_cast<float>(nb_hits) / (static_cast<float>(nb_acting) * wanted_frequency));
        }

        // natural ending (timeout, death)
        for (int i = 0; i < step_result.size(); i++) {
            remaining_frames[i]--;

            if (has_hit[i]) remaining_frames[i] += nb_frames_added_when_hit;
            if (has_kill[i]) remaining_frames[i] += nb_frames_added_when_kill;

            const auto &[state, reward, is_done] = step_result[i];

            if (is_done) {
                if (!already_done[i] && !is_suicide[i]) nb_kills_episode++;
                done[i] = true;
            }

            // starving out (no hit for too long) is a real death: penalized and terminal
            if (!done[i] && remaining_frames[i] <= 0) {
                step_result[i] = {state, reward - 1.f, true};
                done[i] = true;
            }

            if (done[i] && !already_done[i]) {
                const float nb_seconds = static_cast<float>(nb_steps) * wanted_frequency;
                episode_step_mean_nb_metric->add(nb_seconds);
                episode_step_std_nb_metric->add(nb_seconds);
            }
        }

        // detect winner and log reward
        for (int i = 0; i < step_result.size(); i++) {
            if (done[i]) continue;

            // detact winner
            if (const long nb_not_done = std::ranges::count(done, false); nb_not_done == 1) {
                const auto &[state, reward, is_done] = step_result[i];
                if (only_one_tank_alive())
                    step_result[i] = {state, reward + 4.f, true}; // winner réel
                else step_result[i] = {state, reward + 1.f, true};// timeout winner

                done[i] = true;
            }

            // log reward
            reward_metric->add(std::get<1>(step_result[i]));
        }

        nb_steps++;

        return step_result;
    }

    void TrainTankEnvironment::on_draw(
        const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {}

    void TrainTankEnvironment::on_reset_physics(
        const std::unique_ptr<model::AbstractPhysicEngine> &engine) {
        remaining_frames = std::vector(nb_tanks, max_frames_without_hit);

        // close the previous episode's counters (skip the very first reset)
        if (nb_steps > 0) {
            kill_metric->add(static_cast<float>(nb_kills_episode));

            // hits per kill measures how well the damage is concentrated: the theoretical
            // floor is a part's health points. Undefined without a kill — such an episode
            // would inject its raw hit count and swamp the window
            if (nb_kills_episode > 0)
                hits_per_kill_metric->add(
                    static_cast<float>(nb_hits_episode) / static_cast<float>(nb_kills_episode));
        }
        nb_kills_episode = 0;
        nb_hits_episode = 0;

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

    bool TrainTankEnvironment::is_episode_terminated() {
        return are_all_done() || nb_steps > max_episode_steps;
    }

    void TrainTankEnvironment::on_reset_drawables(
        const std::unique_ptr<model::AbstractPhysicEngine> &engine) {}

    void TrainTankEnvironment::reset_singleton() {
        utils::Singleton<utils::Cache<std::shared_ptr<model::Shape>>>::get_singleton()->clear();
        utils::Singleton<utils::Cache<std::shared_ptr<model::Shape>>>::reset_singleton();
    }

    std::vector<std::shared_ptr<AbstractMetric>> TrainTankEnvironment::get_metrics() const {
        return {reward_metric,
                episode_step_mean_nb_metric,
                episode_step_std_nb_metric,
                fire_metric,
                hit_metric,
                kill_metric,
                hits_per_kill_metric};
    }

}// namespace arenai::agent
