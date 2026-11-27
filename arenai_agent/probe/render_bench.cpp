#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include <argparse/argparse.hpp>
#include <sys/resource.h>

#include <arenai_model/shell_economy.h>
#include <arenai_view/backend.h>

#include "./core/train_environment.h"

using namespace arenai;
using clock_type = std::chrono::steady_clock;

namespace {
    double elapsed_ms(const clock_type::time_point start) {
        return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
    }

    long max_rss_mib() {
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        return usage.ru_maxrss / 1024;
    }

    void write_states(std::ofstream &out, const std::vector<core::State> &states) {
        for (const auto &state: states) {
            out.write(
                reinterpret_cast<const char *>(state.vision.pixels.data()),
                static_cast<std::streamsize>(state.vision.pixels.size()));
            out.write(
                reinterpret_cast<const char *>(state.panorama.pixels.data()),
                static_cast<std::streamsize>(state.panorama.pixels.size()));
        }
    }
}// namespace

int main(const int argc, char **argv) {
    argparse::ArgumentParser parser("arenai_agent_render_bench");
    parser.add_argument("--resources_folder").required();
    parser.add_argument("--nb_tanks").scan<'i', int>().default_value(32);
    parser.add_argument("--steps").scan<'i', int>().default_value(300);
    parser.add_argument("--warmup").scan<'i', int>().default_value(20);
    parser.add_argument("--action_repeat").scan<'i', int>().default_value(3);
    parser.add_argument("--vision_num_threads")
        .scan<'i', int>()
        .default_value(static_cast<int>(std::thread::hardware_concurrency()));
    parser.add_argument("--seed").scan<'i', int>().default_value(42);
    parser.add_argument("--frames").default_value(std::string());
    parser.parse_args(argc, argv);

    const int nb_tanks = parser.get<int>("--nb_tanks");
    const int nb_steps = parser.get<int>("--steps");
    const int warmup = parser.get<int>("--warmup");
    constexpr float wanted_frequency = 1.f / 30.f;
    constexpr float spawn_side = 250.f;

    const auto start = clock_type::now();
    auto graphics_backend = view::make_vulkan_backend();
    std::cout << "Vulkan device : " << graphics_backend->renderer_info() << std::endl;

    agent::TrainTankEnvironment env(
        std::move(graphics_backend), nb_tanks,
        std::filesystem::path(parser.get<std::string>("--resources_folder")), wanted_frequency,
        1 << 30, parser.get<int>("--vision_num_threads"), parser.get<int>("--action_repeat"));
    env.seed(static_cast<unsigned int>(parser.get<int>("--seed")));

    const auto reset_states = env.reset(spawn_side, spawn_side, model::GAME_SHELL_ECONOMY);
    const double startup_ms = elapsed_ms(start);

    if (const auto frames_path = parser.get<std::string>("--frames"); !frames_path.empty()) {
        std::ofstream frames(frames_path, std::ios::binary);
        write_states(frames, reset_states);
    }

    std::mt19937 rng(static_cast<unsigned int>(parser.get<int>("--seed")));
    std::uniform_real_distribution axis(-1.f, 1.f);
    std::bernoulli_distribution fire(0.05);

    std::vector<double> step_ms;
    step_ms.reserve(nb_steps);
    for (int step = 0; step < warmup + nb_steps; step++) {
        std::vector<core::Action> actions(nb_tanks);
        for (auto &action: actions)
            action = {
                .left_joystick = {.x = axis(rng), .y = axis(rng)},
                .right_joystick = {.x = axis(rng), .y = axis(rng)},
                .fire_button = {fire(rng)},
                .zoom_button = {false}};

        const auto step_start = clock_type::now();
        env.step(wanted_frequency, actions);
        if (step >= warmup) step_ms.push_back(elapsed_ms(step_start));
    }

    std::ranges::sort(step_ms);
    double sum = 0.0;
    for (const double ms: step_ms) sum += ms;
    const auto percentile = [&](const double p) {
        return step_ms[static_cast<size_t>(p * static_cast<double>(step_ms.size() - 1))];
    };

    std::cout << "startup_ms=" << startup_ms << '\n'
              << "step_mean_ms=" << sum / static_cast<double>(step_ms.size()) << '\n'
              << "step_p50_ms=" << percentile(0.5) << '\n'
              << "step_p90_ms=" << percentile(0.9) << '\n'
              << "step_p99_ms=" << percentile(0.99) << '\n'
              << "max_rss_mib=" << max_rss_mib() << std::endl;

    env.stop_drawing();
    return 0;
}
