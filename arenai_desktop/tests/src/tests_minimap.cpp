#include <cmath>
#include <numbers>
#include <vector>

#include <core/minimap/minimap.h>
#include <core/minimap/minimap_projection.h>
#include <core/minimap/terrain_relief.h>
#include <gtest/gtest.h>
#include <gui/rml/hud.h>

#include <arenai_desktop_tests/headless_rml.h>

using namespace arenai;
using namespace arenai::desktop;

namespace {

    MinimapProjection arena_projection() {
        return {glm::vec2(0.f), 2000.f, Minimap::VIEW_HALF_SIZE};
    }

    TankPose pose(const float x, const float z, const bool alive = true) {
        return {
            .position = glm::vec3(x, 0.f, z), .forward = glm::vec3(0.f, 0.f, -1.f), .alive = alive};
    }

    // left half at grey 0, right half at grey 0.5, the step between columns 20 and 21
    utils::ImageGrey step_image() {
        utils::ImageGrey image{.width = 41, .height = 41, .pixels = std::vector<float>(41 * 41)};
        for (int row = 0; row < image.height; row++)
            for (int col = 21; col < image.width; col++)
                image.pixels[row * image.width + col] = 0.5f;
        return image;
    }

    int red(const gui::MinimapTerrain &terrain, const int i, const int j) {
        return terrain.rgba[(j * terrain.width + i) * 4];
    }

}// namespace

TEST(MinimapProjectionTest, terrain_center_and_corner_map_to_texture_center_and_corner) {
    const MinimapProjection projection = arena_projection();

    const glm::vec2 center = projection.texture_coords(glm::vec3(0.f, 50.f, 0.f));
    EXPECT_FLOAT_EQ(center.x, 0.5f);
    EXPECT_FLOAT_EQ(center.y, 0.5f);

    const glm::vec2 north_east = projection.texture_coords(glm::vec3(2000.f, 0.f, -2000.f));
    EXPECT_FLOAT_EQ(north_east.x, 1.f);
    EXPECT_FLOAT_EQ(north_east.y, 0.f);
}

TEST(MinimapProjectionTest, view_window_is_centered_on_the_player) {
    const auto [view_min, view_max] = arena_projection().view_window(glm::vec3(400.f, 0.f, 0.f));

    // 800 u over 4000 u, centered on x = 0.6
    EXPECT_NEAR(view_min.x, 0.5f, 1e-6f);
    EXPECT_NEAR(view_max.x, 0.7f, 1e-6f);
    EXPECT_NEAR(view_min.y, 0.4f, 1e-6f);
    EXPECT_NEAR(view_max.y, 0.6f, 1e-6f);
}

TEST(MinimapProjectionTest, heading_is_clockwise_from_north) {
    constexpr float pi = std::numbers::pi_v<float>;
    EXPECT_NEAR(MinimapProjection::heading(glm::vec3(0.f, 0.f, -1.f)), 0.f, 1e-6f);
    EXPECT_NEAR(MinimapProjection::heading(glm::vec3(1.f, 0.f, 0.f)), pi / 2.f, 1e-6f);
    EXPECT_NEAR(std::abs(MinimapProjection::heading(glm::vec3(0.f, 0.f, 1.f))), pi, 1e-6f);
    EXPECT_NEAR(MinimapProjection::heading(glm::vec3(0.f, 1.f, 0.f)), 0.f, 1e-6f);
}

TEST(MinimapTest, snapshot_sorts_enemies_into_blips) {
    const Minimap minimap(arena_projection(), {});

    const auto snapshot = minimap.snapshot(
        pose(30.4f, -12.6f), glm::vec3(1.f, -0.2f, 0.f),
        {pose(230.4f, -12.6f), pose(30.4f, 187.4f, false), pose(2000.f, -12.6f),
         pose(-2000.f, -12.6f, false)});

    EXPECT_EQ(snapshot.player_coordinates, glm::ivec2(30, -13));
    EXPECT_NEAR(snapshot.player_heading, 0.f, 1e-6f);
    EXPECT_NEAR(snapshot.view_heading, std::numbers::pi_v<float> / 2.f, 1e-6f);
    ASSERT_EQ(snapshot.blips.size(), 3u);

    EXPECT_EQ(snapshot.blips[0].kind, gui::BlipKind::Enemy);
    EXPECT_NEAR(snapshot.blips[0].position.x, 0.75f, 1e-5f);
    EXPECT_NEAR(snapshot.blips[0].position.y, 0.5f, 1e-5f);

    EXPECT_EQ(snapshot.blips[1].kind, gui::BlipKind::DeadEnemy);
    EXPECT_NEAR(snapshot.blips[1].position.y, 0.75f, 1e-5f);

    EXPECT_EQ(snapshot.blips[2].kind, gui::BlipKind::EnemyOutOfRange);
    EXPECT_NEAR(snapshot.blips[2].position.x, 1.f - Minimap::EDGE_MARGIN, 1e-5f);
    EXPECT_NEAR(snapshot.blips[2].position.y, 0.5f, 1e-5f);
}

TEST(TerrainReliefTest, render_covers_the_extent_opaque) {
    const TerrainRelief relief(step_image(), glm::vec3(10.f, 200.f, 10.f));
    const auto terrain = relief.render(200.f, 2);

    EXPECT_EQ(terrain.width, 80);
    EXPECT_EQ(terrain.height, 80);
    ASSERT_EQ(terrain.rgba.size(), 80u * 80u * 4u);
    for (std::size_t i = 3; i < terrain.rgba.size(); i += 4) ASSERT_EQ(terrain.rgba[i], 255);
}

TEST(TerrainReliefTest, height_step_draws_a_contour_line) {
    const TerrainRelief relief(step_image(), glm::vec3(10.f, 200.f, 10.f));
    const auto terrain = relief.render(200.f, 1);

    // row 5 and columns 5, 6, 20, 25 sit off the 100 u grid
    EXPECT_EQ(red(terrain, 5, 5), red(terrain, 6, 5));
    EXPECT_GT(red(terrain, 20, 5), red(terrain, 25, 5));
}

namespace {

    class NoopOverlayRenderer final : public gui::AbstractOverlayRenderer {
    public:
        void render_overlay() override {}
    };

    class MinimapWidgetTest : public HeadlessRmlTest {};

    bool is_visible(Rml::Element &element) {
        return element.GetComputedValues().visibility() == Rml::Style::Visibility::Visible;
    }

}// namespace

TEST_F(MinimapWidgetTest, update_places_one_blip_per_enemy_and_hides_the_rest) {
    NoopOverlayRenderer renderer;
    gui::HudOverlay hud(renderer);
    hud.load(context());

    hud.minimap().set_terrain({.width = 1, .height = 1, .rgba = {10, 10, 20, 255}});
    hud.minimap().update(
        {.view_min = glm::vec2(0.4f),
         .view_max = glm::vec2(0.6f),
         .player_heading = 0.5f,
         .view_heading = 1.2f,
         .player_coordinates = glm::ivec2(30, -12),
         .blips = {
             {glm::vec2(0.75f, 0.5f), gui::BlipKind::Enemy},
             {glm::vec2(0.5f, 0.75f), gui::BlipKind::DeadEnemy},
             {glm::vec2(0.96f, 0.5f), gui::BlipKind::EnemyOutOfRange}}});
    context().Update();

    Rml::ElementDocument &document = hud.document();
    EXPECT_EQ(document.GetElementById("minimap-coordinates")->GetInnerRML(), "X 30  Z -12");

    Rml::Element *blips = document.GetElementById("minimap-blips");
    ASSERT_EQ(blips->GetNumChildren(), 3);
    EXPECT_FALSE(blips->GetChild(0)->IsClassSet("dead"));
    EXPECT_TRUE(blips->GetChild(1)->IsClassSet("dead"));
    EXPECT_TRUE(blips->GetChild(2)->IsClassSet("edge"));

    hud.minimap().update(
        {.view_min = glm::vec2(0.4f),
         .view_max = glm::vec2(0.6f),
         .blips = {{glm::vec2(0.5f, 0.25f), gui::BlipKind::Enemy}}});
    context().Update();

    ASSERT_EQ(blips->GetNumChildren(), 3);
    EXPECT_TRUE(is_visible(*blips->GetChild(0)));
    EXPECT_FALSE(is_visible(*blips->GetChild(1)));
    EXPECT_FALSE(is_visible(*blips->GetChild(2)));
    EXPECT_EQ(document.GetElementById("minimap-coordinates")->GetInnerRML(), "X 0  Z 0");
}
