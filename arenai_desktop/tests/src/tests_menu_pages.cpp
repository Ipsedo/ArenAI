//
// Created by samuel on 25/10/2026.
//

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
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

    // the data-for clones carry it_index: the window GPU list comes first
    // and reads Auto, GPU A, GPU B
    const auto entries = visible_elements(gui::PageId::Graphics, ".device-entry");
    ASSERT_GE(entries.size(), 3u);
    entries[1]->DispatchEvent(Rml::EventId::Click, Rml::Dictionary());
    EXPECT_EQ(state_.settings.window_gpu, "GPU A");
    entries[0]->DispatchEvent(Rml::EventId::Click, Rml::Dictionary());
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

TEST_F(MenuPagesTest, dpad_reaches_the_cuda_toggle_and_crosses_the_explorer) {
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

    // the toggle sits between the algorithm dropdown and the explorer: the
    // spatial navigation must stop on it on the way down
    ASSERT_TRUE(document.GetElementById("algorithm-head")->Focus(true));
    press(controller::GamepadButton::DPadDown);
    EXPECT_EQ(focused_id(), "cuda-toggle");

    // the explorer bridge hops over the scroll container, and back out of it
    press(controller::GamepadButton::DPadDown);
    const Rml::Element *entry = context().GetFocusElement();
    ASSERT_NE(entry, nullptr);
    EXPECT_TRUE(entry->IsClassSet("file-entry")) << "focus on: " << focused_id();
    press(controller::GamepadButton::DPadUp);
    EXPECT_EQ(focused_id(), "cuda-toggle");

    // and the way up leaves the toggle towards the dropdown
    press(controller::GamepadButton::DPadUp);
    EXPECT_EQ(focused_id(), "algorithm-head");
}

TEST_F(MenuPagesTest, escape_folds_an_open_dropdown_before_leaving_the_page) {
    build_menu(accepting_validator());
    show(gui::PageId::Ai);

    click(gui::PageId::Ai, "toggle_algorithm()");
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
    pause.open(42, [&received](const gui::PopupAction action) { received.push_back(action); });
    EXPECT_TRUE(pause.visible());
    EXPECT_EQ(state_.score, 42);

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
    game_over.open(7, [&nb_actions](gui::PopupAction) { nb_actions++; });
    game_over.on_escape();
    EXPECT_EQ(nb_actions, 0);
}
