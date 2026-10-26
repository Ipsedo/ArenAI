//
// Created by samuel on 29/08/2026.
//

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <core/user_preferences.h>
#include <gtest/gtest.h>

using namespace arenai::desktop;

// ========================================================================
// preferences.json round-trip of the graphics settings, against a store
// redirected to a per-test temp directory (XDG_CACHE_HOME / LOCALAPPDATA)
// ========================================================================

namespace {

#ifdef _WIN32
    constexpr const char *STORE_ENV_VAR = "LOCALAPPDATA";
#else
    constexpr const char *STORE_ENV_VAR = "XDG_CACHE_HOME";
#endif

    void set_env(const char *name, const std::string &value) {
#ifdef _WIN32
        _putenv_s(name, value.c_str());
#else
        setenv(name, value.c_str(), 1);
#endif
    }

    void unset_env(const char *name) {
#ifdef _WIN32
        _putenv_s(name, "");
#else
        unsetenv(name);
#endif
    }

    class UserPreferencesTest : public testing::Test {
    protected:
        void SetUp() override {
            if (const char *previous = std::getenv(STORE_ENV_VAR)) previous_store_ = previous;

            store_dir_ = std::filesystem::path(testing::TempDir())
                         / testing::UnitTest::GetInstance()->current_test_info()->name();
            std::filesystem::remove_all(store_dir_);
            std::filesystem::create_directories(store_dir_);
            set_env(STORE_ENV_VAR, store_dir_.string());
        }

        void TearDown() override {
            if (previous_store_) set_env(STORE_ENV_VAR, *previous_store_);
            else unset_env(STORE_ENV_VAR);
            std::filesystem::remove_all(store_dir_);
        }

        std::filesystem::path store_dir_;
        std::optional<std::string> previous_store_;
    };

}// namespace

TEST_F(UserPreferencesTest, GraphicsSettingsRoundTrip) {
    GameSettings saved;
    saved.shadow_quality = ShadowQuality::Low;
    saved.msaa_samples = 2;
    saved.window_gpu = "NVIDIA GeForce RTX 3060";
    saved.vision_gpu = "AMD Radeon Graphics";

    save_preferences(saved);
    const auto loaded = load_preferences({}, {});

    EXPECT_EQ(loaded.shadow_quality, ShadowQuality::Low);
    EXPECT_EQ(loaded.msaa_samples, 2);
    EXPECT_EQ(loaded.window_gpu, saved.window_gpu);
    EXPECT_EQ(loaded.vision_gpu, saved.vision_gpu);
}

TEST_F(UserPreferencesTest, InvalidGraphicsValuesFallBackToDefaults) {
    const auto path = preferences_path();
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << R"({"shadow_quality": "bogus", "msaa": 3})";

    const auto loaded = load_preferences({}, {});

    EXPECT_EQ(loaded.shadow_quality, GameSettings{}.shadow_quality);
    EXPECT_EQ(loaded.msaa_samples, GameSettings{}.msaa_samples);
}

TEST_F(UserPreferencesTest, MissingGraphicsKeysKeepDefaults) {
    // a preferences.json from before the graphics settings existed
    const auto path = preferences_path();
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << R"({"nb_tanks": 4})";

    const auto loaded = load_preferences({}, {});

    EXPECT_EQ(loaded.nb_tanks, 4);
    EXPECT_EQ(loaded.shadow_quality, GameSettings{}.shadow_quality);
    EXPECT_EQ(loaded.msaa_samples, GameSettings{}.msaa_samples);
    EXPECT_TRUE(loaded.window_gpu.empty());
    EXPECT_TRUE(loaded.vision_gpu.empty());
}

// ========================================================================
// agent preset: the saved name wins over the saved paths when it is a
// known preset, an unknown name leaves a custom selection
// ========================================================================

namespace {

    std::vector<AgentPreset> two_presets() {
        return {
            {.name = "First",
             .algorithm = AiAlgorithm::PpoLiquid,
             .config = "/presets/first/config.json",
             .folder = "/presets/first/save_1"},
            {.name = "Second",
             .algorithm = AiAlgorithm::Ppo,
             .config = "/presets/second/config.json",
             .folder = "/presets/second/save_7"},
        };
    }

}// namespace

TEST_F(UserPreferencesTest, KnownPresetNameRestoresThePresetPaths) {
    GameSettings saved;
    apply_preset(two_presets()[1], saved);
    // the install moved: the paths on disk no longer match the saved ones
    saved.agent_folder = "/old/install/save_7";
    saved.agent_config = "/old/install/config.json";

    save_preferences(saved);
    const auto loaded = load_preferences({}, two_presets());

    EXPECT_EQ(loaded.agent_preset, std::optional<std::string>("Second"));
    EXPECT_EQ(loaded.agent_algorithm, AiAlgorithm::Ppo);
    EXPECT_EQ(loaded.agent_config, std::filesystem::path("/presets/second/config.json"));
    EXPECT_EQ(loaded.agent_folder, std::filesystem::path("/presets/second/save_7"));
}

TEST_F(UserPreferencesTest, CustomSelectionRoundTripsWithoutPreset) {
    GameSettings saved;
    saved.agent_algorithm = AiAlgorithm::PpoLiquid;
    saved.agent_folder = store_dir_;// must exist to be restored
    saved.agent_preset = std::nullopt;

    save_preferences(saved);
    GameSettings defaults;
    apply_preset(two_presets()[0], defaults);
    const auto loaded = load_preferences(defaults, two_presets());

    EXPECT_FALSE(loaded.agent_preset.has_value());
    EXPECT_EQ(loaded.agent_algorithm, AiAlgorithm::PpoLiquid);
    EXPECT_EQ(loaded.agent_folder, store_dir_);
}

TEST_F(UserPreferencesTest, UnknownPresetNameFallsBackToSavedPaths) {
    const auto path = preferences_path();
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << R"({"agent_preset": "Vanished", "algorithm": "ppo"})";

    GameSettings defaults;
    apply_preset(two_presets()[0], defaults);
    const auto loaded = load_preferences(defaults, two_presets());

    EXPECT_FALSE(loaded.agent_preset.has_value());
    EXPECT_EQ(loaded.agent_algorithm, AiAlgorithm::Ppo);
}

TEST_F(UserPreferencesTest, MissingAgentKeysKeepTheDefaultPreset) {
    const auto path = preferences_path();
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << R"({"nb_tanks": 4})";

    GameSettings defaults;
    apply_preset(two_presets()[0], defaults);
    const auto loaded = load_preferences(defaults, two_presets());

    EXPECT_EQ(loaded.agent_preset, std::optional<std::string>("First"));
    EXPECT_EQ(loaded.agent_folder, std::filesystem::path("/presets/first/save_1"));
}
