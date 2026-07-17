//
// Created by samuel on 17/07/2026.
//

#include "./user_preferences.h"

#include <cstdlib>
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

namespace arenai::desktop {

    namespace {

        std::filesystem::path user_cache_dir() {
#ifdef _WIN32
            if (const char *local_app_data = std::getenv("LOCALAPPDATA"))
                return std::filesystem::path(local_app_data) / "ArenAI";
#else
            if (const char *xdg_cache = std::getenv("XDG_CACHE_HOME"))
                return std::filesystem::path(xdg_cache) / "arenai";
            if (const char *home = std::getenv("HOME"))
                return std::filesystem::path(home) / ".cache" / "arenai";
#endif
            // no user profile at all (stripped-down service environment)
            return std::filesystem::temp_directory_path() / "arenai";
        }

    }// namespace

    std::filesystem::path preferences_path() { return user_cache_dir() / "preferences.json"; }

    gui::GameSettings load_preferences(const gui::GameSettings &defaults) {
        auto settings = defaults;

        const auto path = preferences_path();

        try {
            std::ifstream file(path);
            if (!file.is_open()) return settings;// first launch, nothing saved yet

            const auto json = nlohmann::json::parse(file);

            if (const int nb_tanks = json.value("nb_tanks", settings.nb_tanks); nb_tanks > 0)
                settings.nb_tanks = nb_tanks;
            if (const int spawn_side = json.value("spawn_side", settings.spawn_side);
                spawn_side > 0)
                settings.spawn_side = spawn_side;

            settings.controller_kind = json.value("controller", std::string()) == "gamepad"
                                           ? ControllerKind::Gamepad
                                           : ControllerKind::Keyboard;

            // a stale folder (moved, deleted, unplugged drive) falls back to
            // the default so the menu never starts on an unplayable selection
            if (const std::filesystem::path sac_folder = json.value("sac_folder", std::string());
                !sac_folder.empty() && std::filesystem::is_directory(sac_folder))
                settings.sac_folder = sac_folder;
        } catch (const std::exception &e) {
            std::cerr << "Cannot load preferences " << path << ": " << e.what() << std::endl;
            return defaults;
        }

        return settings;
    }

    void save_preferences(const gui::GameSettings &settings) {
        const auto path = preferences_path();

        try {
            const nlohmann::json json = {
                {"nb_tanks", settings.nb_tanks},
                {"spawn_side", settings.spawn_side},
                {"controller",
                 settings.controller_kind == ControllerKind::Gamepad ? "gamepad" : "keyboard"},
                {"sac_folder", settings.sac_folder.string()},
            };

            std::filesystem::create_directories(path.parent_path());

            std::ofstream file(path);
            if (!file.is_open()) throw std::runtime_error("cannot open the file for writing");
            file << json.dump(4) << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Cannot save preferences " << path << ": " << e.what() << std::endl;
        }
    }

}// namespace arenai::desktop
