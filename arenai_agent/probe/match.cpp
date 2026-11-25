#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

#include <arenai_agent/factory.h>
#include <arenai_agent/file_reader.h>
#include <arenai_core/environment.h>
#include <arenai_model/constants.h>
#include <arenai_model/shell_economy.h>
#include <arenai_model/tank_factory.h>
#include <arenai_view/backend.h>

using namespace arenai;

namespace {

    // what one enemy tank did and where it stands, over one game frame
    struct TankFrame {
        bool alive;
        bool out_of_shells;
        bool shells_in_flight;
        float shells_ratio;
        int fired;
        int hits;
        int kills;
        glm::vec3 position;
        // degrees above the horizon
        float canon_elevation;
        // nearest living target (enemy or player) and the one closest to the canon axis
        float nearest_distance;
        // degrees, nearest target seen from the canon axis: horizontal bearing and elevation gap
        float nearest_yaw;
        float nearest_pitch;
        float best_angle;
        float best_distance;
        bool best_is_player;
    };

    struct PlayerFrame {
        bool alive;
        float health_ratio;
        int impacts;
    };

    // the desktop game's rules: no refill of a dry tank, no time limit, the player's tank idle
    class MatchEnvironment final : public core::BaseTanksEnvironment {
    public:
        MatchEnvironment(
            const std::filesystem::path &resources_folder, const int nb_tanks,
            const float wanted_frequency, const int vision_num_threads)
            : BaseTanksEnvironment(
                std::make_shared<agent::DesktopAssetFileReader>(resources_folder),
                view::make_vulkan_backend(), nb_tanks, wanted_frequency, vision_num_threads, false),
              player_dead(false) {}

        // same sequence as DesktopGameEnvironment::play_frame
        std::optional<std::vector<core::State>> play_frame(
            const float time_delta, const std::vector<core::Action> &actions, const bool observe) {
            advance_frame(time_delta, actions);
            render_frame(observe);
            tick_frame();

            if (!player_dead && is_player_hit_to_death()) {
                player_dead = true;
                player_tank->destroy();
            }

            if (!observe) return std::nullopt;
            return read_states();
        }

        PlayerFrame player_frame() const {
            float ratio = 1.f;
            for (const auto &item: player_tank->get_items())
                if (const auto life_item = std::dynamic_pointer_cast<model::LifeItem>(item);
                    life_item)
                    ratio = std::min(ratio, life_item->health_ratio());

            return {
                .alive = !player_dead,
                .health_ratio = ratio,
                .impacts = static_cast<int>(player_tank->consume_received_impacts().size())};
        }

        std::vector<TankFrame> tank_frames() {
            const glm::vec3 player_position(player_tank->get_chassis()->get_model_matrix()[3]);
            const bool player_alive = !player_dead;

            return apply_on_enemies<std::vector<TankFrame>>([&](const auto &tanks) {
                if (scores.empty()) scores.assign(tanks.size(), 0);

                std::vector<glm::vec3> positions;
                positions.reserve(tanks.size());
                for (const auto &tank: tanks)
                    positions.emplace_back(tank->get_chassis()->get_model_matrix()[3]);

                std::vector<TankFrame> frames;
                frames.reserve(tanks.size());

                for (int i = 0; i < static_cast<int>(tanks.size()); i++) {
                    const auto &tank = tanks[i];

                    // score = kills * 10 + hits
                    const int score = tank->get_score();
                    const int gained = score - scores[i];
                    scores[i] = score;

                    const glm::mat4 canon = tank->get_canon()->get_model_matrix();
                    const glm::vec3 origin(canon[3]);
                    const glm::vec3 axis =
                        glm::normalize(glm::mat3(canon) * glm::vec3(0.f, 0.f, 1.f));

                    TankFrame frame{
                        .alive = !tank->is_dead(),
                        .out_of_shells = tank->is_out_of_shells(),
                        .shells_in_flight = tank->has_shells_in_flight(),
                        .shells_ratio = tank->get_remaining_shells_ratio(),
                        .fired = tank->get_last_reward_detail().nb_fired_shells,
                        .hits = gained / 10 + gained % 10,
                        .kills = gained / 10,
                        .position = positions[i],
                        .canon_elevation = std::asin(std::clamp(axis.y, -1.f, 1.f)) * TO_DEGREES,
                        .nearest_distance = std::numeric_limits<float>::infinity(),
                        .nearest_yaw = 0.f,
                        .nearest_pitch = 0.f,
                        .best_angle = 180.f,
                        .best_distance = std::numeric_limits<float>::infinity(),
                        .best_is_player = false};

                    const auto look_at = [&](const glm::vec3 &target, const bool is_player) {
                        const glm::vec3 to_target = target - origin;
                        const float distance = glm::length(to_target);
                        const float angle =
                            std::acos(std::clamp(glm::dot(axis, to_target / distance), -1.f, 1.f))
                            * TO_DEGREES;

                        if (distance < frame.nearest_distance) {
                            frame.nearest_distance = distance;
                            const float yaw =
                                std::atan2(to_target.x, to_target.z) - std::atan2(axis.x, axis.z);
                            frame.nearest_yaw = std::remainder(yaw * TO_DEGREES, 360.f);
                            frame.nearest_pitch =
                                std::asin(std::clamp(to_target.y / distance, -1.f, 1.f))
                                    * TO_DEGREES
                                - frame.canon_elevation;
                        }
                        if (angle < frame.best_angle) {
                            frame.best_angle = angle;
                            frame.best_distance = distance;
                            frame.best_is_player = is_player;
                        }
                    };

                    for (int j = 0; j < static_cast<int>(tanks.size()); j++)
                        if (j != i && !tanks[j]->is_dead()) look_at(positions[j], false);
                    if (player_alive) look_at(player_position, true);

                    frames.push_back(frame);
                }

                return frames;
            });
        }

        // oracle: each living enemy drives its chassis at the nearest living target
        std::vector<core::Action> approach(
            std::vector<core::Action> actions, const float steer, const float throttle,
            const float stop_distance) {
            const glm::vec3 player_position(player_tank->get_chassis()->get_model_matrix()[3]);
            const bool player_alive = !player_dead;

            apply_on_enemies<int>([&](const auto &tanks) {
                for (int i = 0; i < static_cast<int>(tanks.size()); i++) {
                    if (tanks[i]->is_dead()) continue;

                    const glm::mat4 chassis = tanks[i]->get_chassis()->get_model_matrix();
                    glm::vec3 nearest(0.f);
                    float nearest_distance = std::numeric_limits<float>::infinity();

                    const auto look_at = [&](const glm::vec3 &target) {
                        const glm::vec3 local =
                            glm::transpose(glm::mat3(chassis)) * (target - glm::vec3(chassis[3]));
                        if (const float distance = glm::length(local);
                            distance < nearest_distance) {
                            nearest_distance = distance;
                            nearest = local;
                        }
                    };

                    for (int j = 0; j < static_cast<int>(tanks.size()); j++)
                        if (j != i && !tanks[j]->is_dead())
                            look_at(glm::vec3(tanks[j]->get_chassis()->get_model_matrix()[3]));
                    if (player_alive) look_at(player_position);

                    if (!std::isfinite(nearest_distance)) continue;

                    // radians, positive when the target is on the local +x side
                    const float bearing = std::atan2(nearest.x, nearest.z);
                    actions[i].left_joystick.x = std::clamp(steer * bearing, -1.f, 1.f);
                    actions[i].left_joystick.y = nearest_distance > stop_distance ? throttle : 0.f;
                }
                return 0;
            });

            return actions;
        }

        // oracle: the turret slews to the nearest living target while it is out of the field
        std::vector<core::Action>
        acquire(std::vector<core::Action> actions, const float slew, const float half_field) {
            const glm::vec3 player_position(player_tank->get_chassis()->get_model_matrix()[3]);
            const bool player_alive = !player_dead;

            apply_on_enemies<int>([&](const auto &tanks) {
                for (int i = 0; i < static_cast<int>(tanks.size()); i++) {
                    if (tanks[i]->is_dead()) continue;

                    const glm::mat4 canon = tanks[i]->get_canon()->get_model_matrix();
                    const glm::vec3 origin(canon[3]);
                    const glm::vec3 axis = glm::mat3(canon) * glm::vec3(0.f, 0.f, 1.f);
                    glm::vec3 nearest(0.f);
                    float nearest_distance = std::numeric_limits<float>::infinity();

                    const auto look_at = [&](const glm::vec3 &target) {
                        if (const float distance = glm::length(target - origin);
                            distance < nearest_distance) {
                            nearest_distance = distance;
                            nearest = target - origin;
                        }
                    };

                    for (int j = 0; j < static_cast<int>(tanks.size()); j++)
                        if (j != i && !tanks[j]->is_dead())
                            look_at(glm::vec3(tanks[j]->get_chassis()->get_model_matrix()[3]));
                    if (player_alive) look_at(player_position);

                    if (!std::isfinite(nearest_distance)) continue;

                    // degrees, same convention as TankFrame::nearest_yaw
                    const float yaw = std::remainder(
                        (std::atan2(nearest.x, nearest.z) - std::atan2(axis.x, axis.z))
                            * TO_DEGREES,
                        360.f);
                    if (std::abs(yaw) > half_field)
                        actions[i].right_joystick.x = slew * std::copysign(1.f, yaw);
                }
                return 0;
            });

            return actions;
        }

    protected:
        void on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &) override {}

        void on_reset_physics(const std::unique_ptr<model::AbstractPhysicEngine> &engine) override {
            // same spawn as DesktopGameEnvironment
            player_tank = engine->get_tank_factory()->make_player_tank(
                file_reader, "player", glm::vec3(0., -40., 40));
            player_dead = false;
            scores.clear();
        }

        void on_reset_drawables(const std::unique_ptr<model::AbstractPhysicEngine> &) override {}

    private:
        static constexpr float TO_DEGREES = 180.f / std::numbers::pi_v<float>;

        std::unique_ptr<model::PlayerTank> player_tank;
        bool player_dead;
        std::vector<int> scores;

        // the game's defeat test, without the arena exit (the idle player never moves)
        bool is_player_hit_to_death() const {
            return std::ranges::any_of(player_tank->get_items(), [](const auto &item) {
                const auto life_item = std::dynamic_pointer_cast<model::LifeItem>(item);
                return life_item && life_item->is_dead();
            });
        }
    };

    agent::AgentAlgorithm algorithm_from_name(const std::string &name) {
        if (name == "ppo") return agent::PPO;
        if (name == "ppg") return agent::PPG;
        if (name == "ppo_liquid") return agent::PPO_LIQUID;
        throw std::invalid_argument("unknown agent : " + name);
    }

}// namespace

int main(const int argc, char **argv) {
    argparse::ArgumentParser parser("arenai match");

    parser.add_argument("agent");
    parser.add_argument("--run_folder").required();
    parser.add_argument("--save").required();
    parser.add_argument("--resources_folder").required();
    parser.add_argument("--output").required();
    parser.add_argument("--nb_matches").scan<'i', int>().default_value(8);
    parser.add_argument("--nb_tanks").scan<'i', int>().default_value(16);
    parser.add_argument("--spawn_side").scan<'g', float>().default_value(500.f);
    parser.add_argument("--max_seconds").scan<'g', float>().default_value(300.f);
    parser.add_argument("--seed").scan<'i', int>().default_value(0);
    parser.add_argument("--vision_num_threads").scan<'i', int>().default_value(8);
    parser.add_argument("--cuda").default_value(false).implicit_value(true);
    parser.add_argument("--approach_oracle").default_value(false).implicit_value(true);
    parser.add_argument("--approach_steer").scan<'g', float>().default_value(-2.f);
    parser.add_argument("--approach_throttle").scan<'g', float>().default_value(1.f);
    parser.add_argument("--approach_stop_distance").scan<'g', float>().default_value(40.f);

    parser.add_argument("--acquire_oracle").default_value(false).implicit_value(true);
    parser.add_argument("--acquire_slew").scan<'g', float>().default_value(-1.f);
    parser.add_argument("--acquire_half_field").scan<'g', float>().default_value(39.5f);

    parser.parse_args(argc, argv);

    const std::filesystem::path run_folder(parser.get<std::string>("--run_folder"));
    const int nb_matches = parser.get<int>("--nb_matches");
    const int nb_tanks = parser.get<int>("--nb_tanks");
    const float spawn_side = parser.get<float>("--spawn_side");
    const bool acquire_oracle = parser.get<bool>("--acquire_oracle");
    const float acquire_slew = parser.get<float>("--acquire_slew");
    const float acquire_half_field = parser.get<float>("--acquire_half_field");
    const bool approach_oracle = parser.get<bool>("--approach_oracle");
    const float approach_steer = parser.get<float>("--approach_steer");
    const float approach_throttle = parser.get<float>("--approach_throttle");
    const float approach_stop_distance = parser.get<float>("--approach_stop_distance");

    // the game's loading path: deterministic agent built by the public factory
    nlohmann::json config;
    std::ifstream(run_folder / "config.json") >> config;
    agent::AgentFactory factory(config);

    const float wanted_frequency = factory.get_wanted_frequency();
    const int action_repeat = factory.get_action_repeat();
    const int max_frames = static_cast<int>(parser.get<float>("--max_seconds") / wanted_frequency);

    MatchEnvironment env(
        parser.get<std::string>("--resources_folder"), nb_tanks, wanted_frequency,
        parser.get<int>("--vision_num_threads"));

    std::ofstream tanks_csv(parser.get<std::string>("--output") + "_tanks.csv");
    tanks_csv << "match;second;tank;alive;out_of_shells;shells;fired;hits;kills;x;z;"
                 "canon_elevation;nearest_distance;best_angle;best_distance;best_is_player;"
                 "nearest_yaw;nearest_pitch\n";

    std::ofstream player_csv(parser.get<std::string>("--output") + "_player.csv");
    player_csv << "match;second;alive;health;impacts\n";

    std::ofstream matches_csv(parser.get<std::string>("--output") + "_matches.csv");
    matches_csv << "match;seconds;end;enemies_alive;enemies_armed;player_alive\n";

    for (int match = 0; match < nb_matches; match++) {
        env.seed(static_cast<unsigned int>(parser.get<int>("--seed") + match));
        // a fresh agent per match, as a new game session: no recurrent state carried over
        const auto agent = factory.get_agent(
            algorithm_from_name(parser.get<std::string>("agent")), model::ENEMY_STATE_VECTOR_SIZE,
            model::ENEMY_NB_CONTINUOUS_ACTION, model::ENEMY_NB_DISCRETE_ACTION,
            parser.get<bool>("--cuda"));
        agent->load(run_folder / parser.get<std::string>("--save"));

        auto states = env.reset(spawn_side, spawn_side, model::GAME_SHELL_ECONOMY);
        std::vector<core::Action> held_actions;

        // events summed over the frames of one decision
        std::vector<int> fired(nb_tanks, 0), hits(nb_tanks, 0), kills(nb_tanks, 0);
        int impacts = 0;

        std::string end = "time";
        int frame = 0, enemies_alive = nb_tanks, enemies_armed = nb_tanks;
        bool player_alive = true;

        for (; frame < max_frames; frame++) {
            // same decide / observe cadence as GameSession::play_frame
            const bool decide = frame % action_repeat == 0;
            const bool observe = (frame + 1) % action_repeat == 0;

            if (decide) {
                held_actions = agent->act(states);
                if (acquire_oracle)
                    held_actions =
                        env.acquire(std::move(held_actions), acquire_slew, acquire_half_field);
                if (approach_oracle)
                    held_actions = env.approach(
                        std::move(held_actions), approach_steer, approach_throttle,
                        approach_stop_distance);
            }

            if (auto next_states = env.play_frame(
                    wanted_frequency, decide ? held_actions : core::release_fire(held_actions),
                    observe))
                states = std::move(*next_states);

            const auto tank_frames = env.tank_frames();
            const auto player = env.player_frame();

            impacts += player.impacts;
            player_alive = player.alive;
            enemies_alive = 0;
            enemies_armed = 0;

            for (int i = 0; i < nb_tanks; i++) {
                fired[i] += tank_frames[i].fired;
                hits[i] += tank_frames[i].hits;
                kills[i] += tank_frames[i].kills;

                if (!tank_frames[i].alive) continue;
                enemies_alive++;
                if (!tank_frames[i].out_of_shells || tank_frames[i].shells_in_flight)
                    enemies_armed++;
            }

            if (observe) {
                const float second = static_cast<float>(frame + 1) * wanted_frequency;

                for (int i = 0; i < nb_tanks; i++) {
                    const auto &tank = tank_frames[i];
                    tanks_csv << match << ';' << second << ';' << i << ';' << tank.alive << ';'
                              << tank.out_of_shells << ';' << tank.shells_ratio << ';' << fired[i]
                              << ';' << hits[i] << ';' << kills[i] << ';' << tank.position.x << ';'
                              << tank.position.z << ';' << tank.canon_elevation << ';'
                              << tank.nearest_distance << ';' << tank.best_angle << ';'
                              << tank.best_distance << ';' << tank.best_is_player << ';'
                              << tank.nearest_yaw << ';' << tank.nearest_pitch << '\n';
                    fired[i] = hits[i] = kills[i] = 0;
                }

                player_csv << match << ';' << second << ';' << player.alive << ';'
                           << player.health_ratio << ';' << impacts << '\n';
                impacts = 0;
            }

            // nothing can happen any more: no enemy left, or no shell left anywhere
            if (enemies_alive == 0) {
                end = "no_enemy";
                break;
            }
            if (enemies_armed == 0) {
                end = "all_dry";
                break;
            }
        }

        matches_csv << match << ';' << static_cast<float>(frame) * wanted_frequency << ';' << end
                    << ';' << enemies_alive << ';' << enemies_armed << ';' << player_alive << '\n';
        matches_csv.flush();

        std::cout << "match " << match << " : " << static_cast<float>(frame) * wanted_frequency
                  << " s, " << end << ", " << enemies_alive << " alive, " << enemies_armed
                  << " armed, player " << (player_alive ? "alive" : "dead") << std::endl;
    }

    return 0;
}
