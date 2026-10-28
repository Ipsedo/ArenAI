//
// Created by samuel on 25/10/2026.
//

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <gui/rml/decorators/crystals.h>
#include <gui/rml/decorators/facets.h>
#include <gui/rml/hud.h>
#include <gui/rml/input.h>
#include <gui/rml/menu_state.h>
#include <gui/rml/navigator.h>
#include <gui/rml/pages/ai_page.h>
#include <gui/rml/pages/controls_page.h>
#include <gui/rml/pages/graphics_page.h>
#include <gui/rml/pages/main_page.h>
#include <gui/rml/pages/parameters_page.h>
#include <gui/rml/popup_page.h>

#include <arenai_desktop_tests/headless_rml.h>

using namespace arenai;
using namespace arenai::desktop;

// ========================================================================
// The menu screens against a headless RmlUi: navigation through the page
// tree, and the .rml events landing in the shared settings.
// ========================================================================

namespace {

    class MenuPagesTest : public HeadlessRmlTest {
    protected:
        void SetUp() override {
            HeadlessRmlTest::SetUp();

            agent_dir_ = std::filesystem::path(testing::TempDir())
                         / testing::UnitTest::GetInstance()->current_test_info()->name();
            std::filesystem::create_directories(agent_dir_);
            state_.settings.agent_folder = agent_dir_;

            window_ = std::make_shared<FakeWindow>();
            adapter_ = std::make_unique<gui::MenuInputAdapter>(
                &context(), [] {}, [](bool) {});
        }

        void TearDown() override {
            // the context first: destroying the documents detaches the
            // pages' event listeners, which must still exist
            HeadlessRmlTest::TearDown();
            navigator_.reset();
            adapter_.reset();
            std::filesystem::remove_all(agent_dir_);
        }

        // the real page set, with the dry-run validator replaced by `validator`
        void build_menu(AgentValidator validator) {
            std::vector<std::unique_ptr<gui::MenuPage>> pages;
            pages.push_back(std::make_unique<gui::MainPage>(
                state_, [this](const gui::MenuOutcome outcome) { outcome_ = outcome; }));
            pages.push_back(std::make_unique<gui::ParametersPage>(state_));
            pages.push_back(std::make_unique<gui::GraphicsPage>(
                state_, window_, std::vector<std::string>{"GPU A", "GPU B"}));
            pages.push_back(std::make_unique<gui::ControlsPage>(state_, window_, *adapter_));
            pages.push_back(std::make_unique<gui::AiPage>(
                state_, std::vector<AgentPreset>{}, std::move(validator), true));
            navigator_ =
                std::make_unique<gui::Navigator>(context(), std::move(pages), gui::PageId::Main);
            navigator_->show_root();
            context().Update();
        }

        void show(const gui::PageId page) {
            navigator_->go(page);
            context().Update();// builds the data-for clones
        }

        // the visible elements matching `selector` on `page` (data-for keeps
        // its hidden template element in the tree)
        Rml::ElementList visible_elements(const gui::PageId page, const Rml::String &selector) {
            Rml::ElementList all, visible;
            navigator_->page(page).document().QuerySelectorAll(all, selector);
            for (Rml::Element *element: all)
                if (element->IsVisible()) visible.push_back(element);
            return visible;
        }

        void click(const gui::PageId page, const Rml::String &handler) {
            Rml::Element *element = find_event_element(navigator_->page(page).document(), handler);
            ASSERT_NE(element, nullptr) << "no element fires " << handler;
            element->DispatchEvent(Rml::EventId::Click, Rml::Dictionary());
            context().Update();
        }

        bool only_visible(const gui::PageId page) const {
            for (const auto id:
                 {gui::PageId::Main, gui::PageId::Parameters, gui::PageId::Graphics,
                  gui::PageId::Controls, gui::PageId::Ai})
                if (navigator_->page(id).visible() != (id == page)) return false;
            return navigator_->current() == &navigator_->page(page);
        }

        gui::MenuState state_;
        std::optional<gui::MenuOutcome> outcome_;
        std::filesystem::path agent_dir_;
        std::shared_ptr<FakeWindow> window_;
        std::unique_ptr<gui::MenuInputAdapter> adapter_;
        std::unique_ptr<gui::Navigator> navigator_;
    };

    AgentValidator accepting_validator() {
        return [](const AgentSelection &) { return std::nullopt; };
    }

}// namespace

// ========================================================================
// Navigation: the tree is Main > Parameters > {Graphics, Controls, Ai}
// ========================================================================

TEST_F(MenuPagesTest, root_is_the_only_visible_page_at_start) {
    build_menu(accepting_validator());
    EXPECT_TRUE(only_visible(gui::PageId::Main));
}

TEST_F(MenuPagesTest, rml_navigate_events_walk_down_the_tree) {
    build_menu(accepting_validator());

    click(gui::PageId::Main, "navigate('parameters')");
    EXPECT_TRUE(only_visible(gui::PageId::Parameters));

    click(gui::PageId::Parameters, "navigate('graphics')");
    EXPECT_TRUE(only_visible(gui::PageId::Graphics));
}

TEST_F(MenuPagesTest, back_follows_the_declared_parents) {
    build_menu(accepting_validator());
    show(gui::PageId::Ai);

    click(gui::PageId::Ai, "back()");
    EXPECT_TRUE(only_visible(gui::PageId::Parameters));

    navigator_->back();
    EXPECT_TRUE(only_visible(gui::PageId::Main));
}

TEST_F(MenuPagesTest, escape_climbs_the_tree_and_stops_on_the_root) {
    build_menu(accepting_validator());
    show(gui::PageId::Controls);

    EXPECT_TRUE(navigator_->on_escape());
    EXPECT_TRUE(only_visible(gui::PageId::Parameters));
    EXPECT_TRUE(navigator_->on_escape());
    EXPECT_TRUE(only_visible(gui::PageId::Main));
    EXPECT_FALSE(navigator_->on_escape());
    EXPECT_TRUE(only_visible(gui::PageId::Main));
}

TEST_F(MenuPagesTest, hide_all_leaves_no_current_page) {
    build_menu(accepting_validator());
    navigator_->hide_all();
    EXPECT_EQ(navigator_->current(), nullptr);
    for (const auto id: {gui::PageId::Main, gui::PageId::Parameters})
        EXPECT_FALSE(navigator_->page(id).visible());
}

// ========================================================================
// Graphics page: the toggles write the settings
// ========================================================================

TEST_F(MenuPagesTest, graphics_toggles_write_the_settings) {
    build_menu(accepting_validator());
    show(gui::PageId::Graphics);

    click(gui::PageId::Graphics, "set_shadow_quality('low')");
    EXPECT_EQ(state_.settings.shadow_quality, ShadowQuality::Low);

    click(gui::PageId::Graphics, "set_msaa(8)");
    EXPECT_EQ(state_.settings.msaa_samples, 8);

    click(gui::PageId::Graphics, "set_display('fullscreen')");
    EXPECT_TRUE(state_.settings.fullscreen);
    EXPECT_TRUE(window_->fullscreen);
}

TEST_F(MenuPagesTest, graphics_gpu_picker_maps_index_zero_to_auto) {
    state_.settings.window_gpu = "GPU B";
    build_menu(accepting_validator());
    show(gui::PageId::Graphics);

    // the saved name is still listed: it stays
    EXPECT_EQ(state_.settings.window_gpu, "GPU B");

    // unfold the window GPU dropdown: its data-for clones carry it_index and
    // read Auto, GPU A, GPU B (IsVisible ignores the ancestors, so the query
    // stays inside this list: the folded vision one keeps its clones)
    Rml::ElementDocument &document = navigator_->page(gui::PageId::Graphics).document();
    const Rml::Element *list = document.GetElementById("window-gpu-list");
    ASSERT_NE(list, nullptr);
    click(gui::PageId::Graphics, "toggle_window_gpu()");
    EXPECT_TRUE(list->IsVisible());
    const auto entries = visible_elements(gui::PageId::Graphics, "#window-gpu-list .dropdown-item");
    ASSERT_EQ(entries.size(), 3u);
    entries[1]->DispatchEvent(Rml::EventId::Click, Rml::Dictionary());
    context().Update();
    EXPECT_EQ(state_.settings.window_gpu, "GPU A");
    // picking an entry folds the list
    EXPECT_FALSE(list->IsVisible());

    click(gui::PageId::Graphics, "toggle_window_gpu()");
    entries[0]->DispatchEvent(Rml::EventId::Click, Rml::Dictionary());
    context().Update();
    EXPECT_TRUE(state_.settings.window_gpu.empty());
}

// ========================================================================
// AI page: the dry-run validator gates Play
// ========================================================================

TEST_F(MenuPagesTest, valid_agent_opens_the_play_gate) {
    int nb_checks = 0;
    build_menu([&nb_checks, this](const AgentSelection &selection) {
        nb_checks++;
        EXPECT_EQ(selection.folder, agent_dir_);
        return std::nullopt;
    });

    EXPECT_EQ(nb_checks, 1);
    EXPECT_TRUE(state_.agent_valid);

    click(gui::PageId::Main, "play()");
    ASSERT_TRUE(outcome_.has_value());
    EXPECT_EQ(*outcome_, gui::MenuOutcome::Play);
}

TEST_F(MenuPagesTest, rejected_agent_keeps_play_closed) {
    build_menu([](const AgentSelection &) { return std::string("state dict missing"); });

    EXPECT_FALSE(state_.agent_valid);

    click(gui::PageId::Main, "play()");
    EXPECT_FALSE(outcome_.has_value());

    click(gui::PageId::Main, "exit()");
    ASSERT_TRUE(outcome_.has_value());
    EXPECT_EQ(*outcome_, gui::MenuOutcome::Quit);
}

TEST_F(MenuPagesTest, switching_algorithm_reruns_the_check_and_detaches_the_preset) {
    std::vector<AiAlgorithm> checked;
    state_.settings.agent_preset = "unknown preset";
    build_menu([&checked](const AgentSelection &selection) {
        checked.push_back(selection.algorithm);
        return std::nullopt;
    });
    show(gui::PageId::Ai);

    click(gui::PageId::Ai, "set_algorithm('ppo_liquid')");
    EXPECT_EQ(state_.settings.agent_algorithm, AiAlgorithm::PpoLiquid);
    EXPECT_FALSE(state_.settings.agent_preset.has_value());
    ASSERT_EQ(checked.size(), 2u);
    EXPECT_EQ(checked.back(), AiAlgorithm::PpoLiquid);
}

TEST_F(MenuPagesTest, dpad_reaches_the_device_toggles_and_crosses_the_explorer) {
    build_menu(accepting_validator());
    show(gui::PageId::Ai);

    Rml::ElementDocument &document = navigator_->page(gui::PageId::Ai).document();
    const auto press = [this](const controller::GamepadButton button) {
        adapter_->on_gamepad_button(button, controller::InputAction::Press);
    };
    const auto focused_id = [this]() -> Rml::String {
        const Rml::Element *focus = context().GetFocusElement();
        return focus != nullptr ? focus->GetId() : "<none>";
    };

    // the device toggles sit right under the algorithm grid: the spatial
    // navigation must stop on them on the way down from its bottom-left chip
    ASSERT_TRUE(document.GetElementById("algo-ppo-impala")->Focus(true));
    press(controller::GamepadButton::DPadDown);
    EXPECT_EQ(focused_id(), "cpu-toggle");

    // the explorer bridge hops over the scroll container from either
    // toggle, and lands back on the CUDA one on the way out
    press(controller::GamepadButton::DPadDown);
    const Rml::Element *entry = context().GetFocusElement();
    ASSERT_NE(entry, nullptr);
    EXPECT_TRUE(entry->IsClassSet("file-entry")) << "focus on: " << focused_id();
    press(controller::GamepadButton::DPadUp);
    EXPECT_EQ(focused_id(), "cuda-toggle");

    // and the way up leaves the toggle towards the algorithm grid
    press(controller::GamepadButton::DPadUp);
    EXPECT_EQ(focused_id(), "algo-ppo-liquid-impala");
}

TEST_F(MenuPagesTest, escape_folds_an_open_dropdown_before_leaving_the_page) {
    build_menu(accepting_validator());
    show(gui::PageId::Ai);

    click(gui::PageId::Ai, "toggle_preset()");
    EXPECT_TRUE(navigator_->on_escape());
    EXPECT_TRUE(only_visible(gui::PageId::Ai));

    EXPECT_TRUE(navigator_->on_escape());
    EXPECT_TRUE(only_visible(gui::PageId::Parameters));
}

// ========================================================================
// Controls page: a binding capture goes through the input adapter
// ========================================================================

TEST_F(MenuPagesTest, keyboard_capture_rebinds_the_slot) {
    build_menu(accepting_validator());
    show(gui::PageId::Controls);

    // the capturable chips, in slot order (the fixed rows have none)
    const auto chips = visible_elements(gui::PageId::Controls, ".keychip[data-event-click]");
    ASSERT_FALSE(chips.empty());

    // slot 0 is FORWARD: click its chip, then press F
    chips[0]->DispatchEvent(Rml::EventId::Click, Rml::Dictionary());
    adapter_->on_key(controller::Key::F, controller::InputAction::Press);

    ASSERT_TRUE(state_.settings.bindings.keyboard.forward.has_value());
    EXPECT_EQ(
        std::get<controller::Key>(*state_.settings.bindings.keyboard.forward), controller::Key::F);
}

TEST_F(MenuPagesTest, leaving_the_controls_page_cancels_a_pending_capture) {
    build_menu(accepting_validator());
    show(gui::PageId::Controls);

    const auto chips = visible_elements(gui::PageId::Controls, ".keychip[data-event-click]");
    ASSERT_FALSE(chips.empty());
    chips[0]->DispatchEvent(Rml::EventId::Click, Rml::Dictionary());

    navigator_->back();
    // the capture sink is gone: the key reaches nobody and the slot keeps W
    adapter_->on_key(controller::Key::F, controller::InputAction::Press);
    EXPECT_EQ(
        std::get<controller::Key>(*state_.settings.bindings.keyboard.forward), controller::Key::W);
}

// ========================================================================
// Popups: the listener receives the player's choice
// ========================================================================

TEST_F(MenuPagesTest, popup_reports_actions_to_its_listener) {
    gui::PopupPage pause(gui::PopupKind::Pause, state_);
    pause.load(context());

    std::vector<gui::PopupAction> received;
    pause.open(
        {.elapsed_seconds = 61.f,
         .player_score = 42,
         .enemies_mean_score = 7.6f,
         .enemies_alive = 3},
        [&received](const gui::PopupAction action) { received.push_back(action); });
    context().Update();
    EXPECT_TRUE(pause.visible());
    EXPECT_EQ(state_.match.player_score, 42);
    EXPECT_EQ(pause.document().GetElementById("pause-clock")->GetInnerRML(), "01:01");
    EXPECT_EQ(pause.document().GetElementById("pause-machine-score")->GetInnerRML(), "8");

    Rml::Element *main_menu = find_event_element(pause.document(), "popup_main_menu()");
    ASSERT_NE(main_menu, nullptr);
    main_menu->DispatchEvent(Rml::EventId::Click, Rml::Dictionary());
    pause.on_escape();

    ASSERT_EQ(received.size(), 2u);
    EXPECT_EQ(received[0], gui::PopupAction::MainMenu);
    EXPECT_EQ(received[1], gui::PopupAction::Continue);

    pause.close();
    EXPECT_FALSE(pause.visible());
}

TEST_F(MenuPagesTest, game_over_popup_ignores_escape) {
    gui::PopupPage game_over(gui::PopupKind::GameOver, state_);
    game_over.load(context());

    int nb_actions = 0;
    game_over.open({.player_score = 7}, [&nb_actions](gui::PopupAction) { nb_actions++; });
    game_over.on_escape();
    EXPECT_EQ(nb_actions, 0);
}

// ========================================================================
// Backdrops: the facets declarations survive the RCSS shorthand parsing
// (quoted polygon strings inside a decorator list)
// ========================================================================

TEST_F(MenuPagesTest, main_menu_backdrop_declares_its_facets) {
    build_menu(accepting_validator());

    Rml::Element *body = &navigator_->page(gui::PageId::Main).document();
    const Rml::Property *property = body->GetProperty(Rml::PropertyId::Decorator);
    ASSERT_NE(property, nullptr);
    const auto decorators = property->Get<Rml::DecoratorsPtr>();
    ASSERT_NE(decorators, nullptr);
    // two ground halves, two fill passes, two line passes, the seam dashes,
    // the center lozenges, two crystal camps
    ASSERT_EQ(decorators->list.size(), 10u);

    int nb_houle = 0, nb_crystals = 0;
    for (const Rml::DecoratorDeclaration &declaration: decorators->list) {
        ASSERT_NE(declaration.instancer, nullptr);
        const Rml::PropertySpecification &spec = declaration.instancer->GetPropertySpecification();

        if (declaration.type == "crystals") {
            ++nb_crystals;
            const Rml::PropertyDefinition *instances = spec.GetProperty("instances");
            ASSERT_NE(instances, nullptr);
            const Rml::Property *value = declaration.properties.GetProperty(instances->GetId());
            ASSERT_NE(value, nullptr);
            EXPECT_FALSE(gui::CrystalsDecorator::parse(value->Get<Rml::String>()).empty())
                << "instances: " << value->Get<Rml::String>();
            continue;
        }

        EXPECT_EQ(declaration.type, "facets");
        const Rml::PropertyDefinition *polygons = spec.GetProperty("polygons");
        ASSERT_NE(polygons, nullptr);
        const Rml::Property *value = declaration.properties.GetProperty(polygons->GetId());
        ASSERT_NE(value, nullptr);
        const auto parsed = gui::FacetsDecorator::parse(value->Get<Rml::String>());
        EXPECT_FALSE(parsed.empty()) << "polygons: " << value->Get<Rml::String>();

        const Rml::PropertyDefinition *animation = spec.GetProperty("animation");
        ASSERT_NE(animation, nullptr);
        const Rml::Property *mode = declaration.properties.GetProperty(animation->GetId());
        if (mode != nullptr && mode->Get<Rml::String>() == "houle") ++nb_houle;
    }
    // the colored layers and the lozenges swell; grounds and dashes stay
    // still; each camp tumbles its own crystals
    EXPECT_EQ(nb_houle, 5);
    EXPECT_EQ(nb_crystals, 2);
}

TEST_F(MenuPagesTest, crystals_parse_reads_instances) {
    const auto instances =
        gui::CrystalsDecorator::parse("165 255 82 1 1 0|985 300 38 2 -1.3 1.3|bogus");
    ASSERT_EQ(instances.size(), 2u);
    EXPECT_FLOAT_EQ(instances[0].center.x, 165.f);
    EXPECT_FLOAT_EQ(instances[0].size, 82.f);
    EXPECT_EQ(instances[0].shape, 1);
    EXPECT_EQ(instances[1].shape, 2);
    EXPECT_FLOAT_EQ(instances[1].spin, -1.3f);
    EXPECT_FLOAT_EQ(instances[1].phase, 1.3f);
}

TEST_F(MenuPagesTest, facets_parse_reads_points_and_opacity) {
    const auto polygons = gui::FacetsDecorator::parse("0 0 10 0 10 10@0.5|640 0 640 6|bogus");
    ASSERT_EQ(polygons.size(), 2u);
    EXPECT_EQ(polygons[0].points.size(), 3u);
    EXPECT_FLOAT_EQ(polygons[0].opacity, 0.5f);
    EXPECT_EQ(polygons[1].points.size(), 2u);
    EXPECT_FLOAT_EQ(polygons[1].opacity, 1.f);
}

// ========================================================================
// The asymmetric swell: anchored at the horizon and the screen edges,
// different on the two halves of the arena, and alive over time
// ========================================================================

TEST_F(MenuPagesTest, facets_swell_is_anchored_at_horizon_and_edges) {
    for (const double t: {0.0, 1.7, 42.0}) {
        EXPECT_FLOAT_EQ(gui::FacetsDecorator::swell_offset({320.f, 720.f}, t), 0.f);
        EXPECT_NEAR(gui::FacetsDecorator::swell_offset({320.f, 395.f}, t), 0.f, 1e-3f);
        EXPECT_NEAR(gui::FacetsDecorator::swell_offset({320.f, 0.f}, t), 0.f, 1e-3f);
    }
}

TEST_F(MenuPagesTest, facets_swell_moves_the_floor_over_time) {
    const Rml::Vector2f mid_floor{320.f, 650.f};
    const float at_rest = gui::FacetsDecorator::swell_offset(mid_floor, 0.0);
    float max_travel = 0.f;
    for (const double t: {1.0, 2.0, 3.0})
        max_travel = std::max(
            max_travel, std::abs(gui::FacetsDecorator::swell_offset(mid_floor, t) - at_rest));
    EXPECT_GT(max_travel, 1.f);
}

TEST_F(MenuPagesTest, facets_swell_is_asymmetric_between_the_two_sides) {
    // mirrored probes: were the two halves driven by one symmetric field,
    // these would stay equal at every time
    float max_difference = 0.f;
    for (const double t: {0.0, 0.5, 1.0, 1.5}) {
        const float player = gui::FacetsDecorator::swell_offset({320.f, 650.f}, t);
        const float machine = gui::FacetsDecorator::swell_offset({960.f, 650.f}, t);
        max_difference = std::max(max_difference, std::abs(player - machine));
    }
    EXPECT_GT(max_difference, 1.f);
}

// ========================================================================
// HUD: the match status lands in the bound elements
// ========================================================================

namespace {
    class NoopOverlayRenderer final : public gui::AbstractOverlayRenderer {
    public:
        void render_overlay() override {}
    };
}// namespace

TEST_F(MenuPagesTest, hud_status_fills_clock_scores_hull_and_enemy_slots) {
    NoopOverlayRenderer renderer;
    gui::HudOverlay hud(renderer);
    hud.load(context());

    hud.set_status(
        {.elapsed_seconds = 252.6f,
         .player_score = 1240,
         .player_health = 0.7f,
         .enemies_mean_score = 859.6f,
         .enemies_alive = 3,
         .enemies_total = 4});
    context().Update();

    Rml::ElementDocument &document = hud.document();
    EXPECT_EQ(document.GetElementById("hud-clock")->GetInnerRML(), "04:12");
    EXPECT_EQ(document.GetElementById("hud-player-score")->GetInnerRML(), "1240");
    EXPECT_EQ(document.GetElementById("hud-ai-score")->GetInnerRML(), "860");
    EXPECT_EQ(document.GetElementById("hud-enemies-left")->GetInnerRML(), "3 / 4");

    // 0.7 of 8 segments: 6 kept (ceil), 2 lost
    Rml::ElementList lost;
    document.QuerySelectorAll(lost, ".hull-segment.lost");
    EXPECT_EQ(lost.size(), 2u);
}
