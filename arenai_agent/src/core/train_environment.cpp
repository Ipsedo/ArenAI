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
        const int vision_num_threads, const int action_repeat)
        : BaseTanksEnvironment(
            std::make_shared<DesktopAssetFileReader>(android_assets_path), graphics_backend,
            nb_tanks, wanted_frequency, vision_height, vision_width, vision_num_threads, false,
            true),
          wanted_frequency(wanted_frequency), action_repeat(action_repeat), nb_tanks(nb_tanks),
          nb_steps(0), done(nb_tanks, false), already_done(nb_tanks, false),
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
          hit_metric(std::make_shared<MeanMetric>("hit", 16, 2, true)),
          kill_metric(std::make_shared<MeanMetric>("kill", 16, 1)), nb_kills_episode(0),
          nb_fires_episode(0), nb_hits_episode(0) {}

    std::vector<std::tuple<core::State, core::Reward, core::IsDone, core::IsTruncated>>
    TrainTankEnvironment::step(const float time_delta, const std::vector<core::Action> &actions) {

        // tanks flagged done on a previous step already emitted their terminal transition:
        // mark them so the caller can skip their post-mortem steps
        already_done = done;

        // one agent decision spans action_repeat physics frames; the window's rewards
        // are summed, a death or a win on any of its frames ends the tank's transition
        std::vector rewards(nb_tanks, 0.f);
        std::vector truncated(nb_tanks, false);
        std::vector<model::RewardDetail> reward_details(nb_tanks);
        std::vector nb_fires_per_tank(nb_tanks, 0);
        std::vector nb_hits_per_tank(nb_tanks, 0);

        int nb_frames = 0;

        for (int frame = 0; frame < action_repeat && !is_episode_terminated(); frame++) {
            advance_frame(time_delta, frame == 0 ? actions : core::release_fire(actions));

            // the agents' visions are rendered once, after the window
            const auto frame_results = tick_frame();

            nb_frames++;
            nb_steps++;

            // the tanks' events are consumed every frame: tick() reads them
            apply_on_enemies<int>([&](const auto &tanks) {
                for (int i = 0; i < nb_tanks; i++) {
                    const bool has_hit = tanks[i]->consume_has_hit();
                    const bool has_kill = tanks[i]->consume_has_kill();
                    const bool has_fired = tanks[i]->consume_has_fire();

                    // a tank's rewards stop at its death frame
                    if (done[i]) continue;

                    const auto &[reward, is_done, is_truncated] = frame_results[i];

                    rewards[i] += reward;
                    truncated[i] = is_truncated;

                    const auto detail = tanks[i]->get_last_reward_detail();
                    reward_details[i].aim += detail.aim;
                    reward_details[i].hit += detail.hit;
                    reward_details[i].received += detail.received;
                    reward_details[i].death += detail.death;
                    reward_details[i].nb_landed_shells += detail.nb_landed_shells;
                    reward_details[i].sum_aim_quality += detail.sum_aim_quality;
                    reward_details[i].sum_miss_distance += detail.sum_miss_distance;

                    if (has_hit) {
                        nb_hits_per_tanks[i]++;
                        nb_hits_per_tank[i]++;
                    }
                    if (has_kill) nb_kills_per_tanks[i]++;
                    if (has_fired) nb_fires_per_tank[i]++;

                    // detect death (kill, suicide or timeout)
                    if (is_done) {
                        if (!tanks[i]->is_suicide() && !tanks[i]->is_timeout()) nb_kills_episode++;
                        done[i] = true;
                    }
                }
                return 0;
            });

            // detect winner
            std::vector<int> tanks_not_done_indexes;
            for (int i = 0; i < nb_tanks; i++)
                if (!done[i]) tanks_not_done_indexes.push_back(i);

            if (tanks_not_done_indexes.size() == 1) {
                const auto winner_index = tanks_not_done_indexes[0];

                rewards[winner_index] += nb_kills_per_tanks[winner_index] > 0 ? 2.f : 0.f;
                // winning is a genuine termination, never a truncation
                truncated[winner_index] = false;
                done[winner_index] = true;
            }
        }

        render_frame(true);
        auto states = read_states();

        // fire frequency (per second, per tank that acted this step)
        int nb_acting = 0, nb_fires = 0, nb_hits = 0;
        for (int i = 0; i < nb_tanks; i++) {
            if (already_done[i]) continue;
            nb_acting++;
            nb_fires += nb_fires_per_tank[i];
            nb_hits += nb_hits_per_tank[i];
        }

        if (nb_acting > 0 && nb_frames > 0)
            fire_metric->add(
                static_cast<float>(nb_fires)
                / (static_cast<float>(nb_acting * nb_frames) * wanted_frequency));

        nb_fires_episode += nb_fires;
        nb_hits_episode += nb_hits;

        std::vector<std::tuple<core::State, core::Reward, core::IsDone, core::IsTruncated>> result;
        result.reserve(nb_tanks);

        // log metrics
        for (int i = 0; i < nb_tanks; i++) {
            if (!already_done[i]) {
                reward_metric->add(rewards[i]);

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

            result.emplace_back(std::move(states[i]), rewards[i], done[i], truncated[i]);
        }

        return result;
    }

    void TrainTankEnvironment::on_draw(
        const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {}

    void TrainTankEnvironment::on_reset_physics(
        const std::unique_ptr<model::AbstractPhysicEngine> &engine) {

        // close the previous episode's counters (skip the very first reset)
        if (nb_steps > 0) {
            kill_metric->add(static_cast<float>(nb_kills_episode));

            // hit accuracy is undefined on an episode without a single fire
            if (nb_fires_episode > 0)
                hit_metric->add(
                    static_cast<float>(nb_hits_episode) / static_cast<float>(nb_fires_episode));
        }
        nb_kills_episode = 0;

        nb_steps = 0;

        already_done = std::vector(nb_tanks, false);
        done = std::vector(nb_tanks, false);

        nb_hits_per_tanks = std::vector(nb_tanks, 0);
        nb_kills_per_tanks = std::vector(nb_tanks, 0);

        nb_fires_episode = 0;
        nb_hits_episode = 0;
    }

    int TrainTankEnvironment::episode_nb_fires() const { return nb_fires_episode; }

    int TrainTankEnvironment::episode_nb_hits() const { return nb_hits_episode; }

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
