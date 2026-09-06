//
// Created by samuel on 11/03/2026.
//

#include <filesystem>

#include <argparse/argparse.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

#include "./game.h"

using namespace arenai;
using namespace arenai::desktop;

static std::filesystem::path executable_dir(const char *argv0) {
#ifdef _WIN32
    wchar_t buffer[MAX_PATH];
    if (GetModuleFileNameW(nullptr, buffer, MAX_PATH) > 0)
        return std::filesystem::path(buffer).parent_path();
#else
    std::error_code error;
    const auto exe_path = std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error) return exe_path.parent_path();
#endif
    return std::filesystem::absolute(argv0).parent_path();
}

int main(const int argc, char **argv) {
    argparse::ArgumentParser parser("arenai game");

    // everything about the AI model (vision size, frequency, hyper-parameters)
    // comes from the config.json selected in the menu; the command line only
    // keeps what belongs to the player's machine
    parser.add_argument("--window_width").scan<'i', int>().default_value(1920);
    parser.add_argument("--window_height").scan<'i', int>().default_value(1080);
    parser.add_argument("--cuda").implicit_value(true).default_value(false);

    parser.parse_args(argc, argv);

    const auto resources_folder = executable_dir(argv[0]) / "resources";

    run_gui(
        {.window_width = parser.get<int>("--window_width"),
         .window_height = parser.get<int>("--window_height"),
         .resources_folder = resources_folder},
        {.state_dict_folder = resources_folder / "trained_models" / "ppo_train_386" / "save_66",
         .config_json = resources_folder / "trained_models" / "ppo_train_386" / "config.json",
         .cuda = parser.get<bool>("--cuda")});

    return 0;
}
