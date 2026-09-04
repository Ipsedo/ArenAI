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
            nb_tanks, wanted_frequency, vision_height, vision_width, vision_num_threads, false,
            true, static_cast<float>(max_episode_steps) * wanted_frequency),
          wanted_frequency(wanted_frequency), nb_tanks(nb_tanks), nb_steps(0),
          done(nb_tanks, false), already_done(nb_tanks, false),
          max_episode_steps(max_episode_steps), nb_hits_per_tanks(nb_tanks, 0),
          nb_kills_per_tanks(nb_tanks, 0), reward_metric(std::make_shared<MeanMetric>(
                                               "r", 4 * nb_tanks * max_episode_steps, 1, true)),
          reward_aim_metric(
              std::make_shared<MeanMetric>("r_aim", 4 * nb_tanks * max_episode_steps, 1, true)),
          reward_hit_metric(
              std::make_shared<MeanMetric>("r_hit", 4 * nb_tanks * max_episode_steps, 1, true)),
          reward_received_metric(
              std::make_shared<MeanMetric>("r_rcv", 4 * nb_tanks * max_episode_steps, 1, true)),
          miss_distance_metric(std::make_shared<MeanMetric>("miss", 1024 * nb_tanks, 1)),
          episode_step_mean_nb_metric(std::make_shared<MeanMetric>("s", 32, 1)),
          fire_metric(std::make_shared<MeanMetric>("fire", 256, 2)),
          hit_metric(std::make_shared<MeanMetric>("hit", 256, 2, true)),
          kill_metric(std::make_shared<MeanMetric>("kill", 16, 1)), nb_kills_episode(0) {}

    std::vector<std::tuple<core::State, core::Reward, core::IsDone>>
    TrainTankEnvironment::step(const float time_delta, const std::vector<core::Action> &actions) {

        // tanks flagged done on a previous step already emitted their terminal transition:
        // mark them so the caller can skip their post-mortem steps
        already_done = done;

        auto step_result = BaseTanksEnvironment::step(time_delta, actions);

        const auto has_hit = apply_on_enemies<std::vector<bool>>([&](const auto &factories) {
            std::vector<bool> has_hit_result;
            has_hit_result.reserve(nb_tanks);
            for (const auto &factory: factories)
                has_hit_result.push_back(factory->consume_has_hit());
            return has_hit_result;
        });

        const auto has_kill = apply_on_enemies<std::vector<bool>>([&](const auto &factories) {
            std::vector<bool> has_kill_result;
            has_kill_result.reserve(nb_tanks);
            for (const auto &factory: factories)
                has_kill_result.push_back(factory->consume_has_kill());
            return has_kill_result;
        });

        const auto has_fired = apply_on_enemies<std::vector<bool>>([&](const auto &factories) {
            std::vector<bool> has_fired_result;
            has_fired_result.reserve(nb_tanks);
            for (const auto &factory: factories)
                has_fired_result.push_back(factory->consume_has_fire());
            return has_fired_result;
        });

        const auto reward_details =
            apply_on_enemies<std::vector<model::RewardDetail>>([&](const auto &factories) {
                std::vector<model::RewardDetail> details;
                details.reserve(nb_tanks);
                for (const auto &factory: factories)
                    details.push_back(factory->get_last_reward_detail());
                return details;
            });

        const auto is_suicide = apply_on_enemies<std::vector<bool>>([&](const auto &factories) {
            std::vector<bool> is_suicide_result;
            is_suicide_result.reserve(nb_tanks);
            for (const auto &factory: factories) is_suicide_result.push_back(factory->is_suicide());
            return is_suicide_result;
        });

        const auto is_timeout = apply_on_enemies<std::vector<bool>>([&](const auto &factories) {
            std::vector<bool> is_timeout_result;
            is_timeout_result.reserve(nb_tanks);
            for (const auto &factory: factories) is_timeout_result.push_back(factory->is_timeout());
            return is_timeout_result;
        });

        // fire / hit frequencies (per second, per tank that acted this step)
        int nb_acting = 0, nb_fires = 0, nb_hits = 0;
        for (int i = 0; i < nb_tanks; i++) {
            if (already_done[i]) continue;
            nb_acting++;
            nb_fires += has_fired[i] ? 1 : 0;
            nb_hits += has_hit[i] ? 1 : 0;
        }

        if (nb_acting > 0) {
            fire_metric->add(
                static_cast<float>(nb_fires) / (static_cast<float>(nb_acting) * wanted_frequency));
            hit_metric->add(
                static_cast<float>(nb_hits) / (static_cast<float>(nb_acting) * wanted_frequency));
        }

        // step over tanks (hits and kills counters)
        for (int i = 0; i < step_result.size(); i++) {

            if (has_hit[i]) { nb_hits_per_tanks[i] += 1; }
            if (has_kill[i]) { nb_kills_per_tanks[i] += 1; }

            // detect death (kill, suicide or timeout)
            if (const auto &[state, reward, is_done] = step_result[i]; is_done) {
                if (!already_done[i] && !is_suicide[i] && !is_timeout[i]) nb_kills_episode++;
                done[i] = true;
            }
        }

        // detect winner
        std::vector<int> tanks_not_done_indexes;
        for (int i = 0; i < nb_tanks; i++)
            if (!done[i]) tanks_not_done_indexes.push_back(i);

        if (tanks_not_done_indexes.size() == 1) {
            const auto winner_index = tanks_not_done_indexes[0];

            const auto &[state, reward, is_done] = step_result[winner_index];

            constexpr float win_reward = 2.f;
            step_result[winner_index] = {state, reward + win_reward, true};
            done[winner_index] = true;
        }

        // log metrics
        for (int i = 0; i < step_result.size(); i++)
            if (!already_done[i]) {
                reward_metric->add(std::get<1>(step_result[i]));

                const auto &detail = reward_details[i];

                reward_aim_metric->add(detail.aim);
                reward_hit_metric->add(detail.hit);
                reward_received_metric->add(detail.received);

                // undefined on a step where no shell landed: averaging a zero in would
                // measure the firing rate, not the aim
                if (detail.nb_landed_shells > 0)
                    miss_distance_metric->add(
                        detail.sum_miss_distance / static_cast<float>(detail.nb_landed_shells));

                if (done[i])
                    episode_step_mean_nb_metric->add(
                        static_cast<float>(nb_steps) * wanted_frequency);
            }

        nb_steps++;

        return step_result;
    }

    void TrainTankEnvironment::on_draw(
        const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {}

    void TrainTankEnvironment::on_reset_physics(
        const std::unique_ptr<model::AbstractPhysicEngine> &engine) {

        // close the previous episode's counter (skip the very first reset)
        if (nb_steps > 0) kill_metric->add(static_cast<float>(nb_kills_episode));
        nb_kills_episode = 0;

        nb_steps = 0;

        already_done = std::vector(nb_tanks, false);
        done = std::vector(nb_tanks, false);

        nb_hits_per_tanks = std::vector(nb_tanks, 0);
        nb_kills_per_tanks = std::vector(nb_tanks, 0);
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
        return {reward_metric,        reward_aim_metric,
                reward_hit_metric,    reward_received_metric,
                miss_distance_metric, episode_step_mean_nb_metric,
                fire_metric,          hit_metric,
                kill_metric};
    }

}// namespace arenai::agent
