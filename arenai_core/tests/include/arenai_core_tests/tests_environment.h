#ifndef ARENAI_CORE_TESTS_ENVIRONMENT_H
#define ARENAI_CORE_TESTS_ENVIRONMENT_H

#include <gtest/gtest.h>

#include <arenai_core/environment.h>

#include "./utils/engine_test_fixture.h"

using namespace arenai;

class TestTanksEnvironment final : public core::BaseTanksEnvironment {
public:
    using BaseTanksEnvironment::BaseTanksEnvironment;
    using BaseTanksEnvironment::respawn_tank;

    glm::vec3 chassis_position(const int index) {
        return apply_on_enemies<glm::vec3>([index](const auto &tanks) {
            return glm::vec3(tanks[index]->get_chassis()->get_model_matrix()[3]);
        });
    }

    bool out_of_shells(const int index) {
        return apply_on_enemies<bool>(
            [index](const auto &tanks) { return tanks[index]->is_out_of_shells(); });
    }

    int draw_call_count = 0;
    int reset_physics_call_count = 0;
    int reset_drawables_call_count = 0;

    std::vector<std::pair<std::string, glm::vec3>> spawn_positions;

protected:
    void on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override {
        draw_call_count++;
    }

    void on_reset_physics(const std::unique_ptr<model::AbstractPhysicEngine> &engine) override {
        reset_physics_call_count++;

        spawn_positions.clear();
        for (const auto &item: engine->get_items())
            spawn_positions.emplace_back(item->get_name(), glm::vec3(item->get_model_matrix()[3]));
    }

    void on_reset_drawables(const std::unique_ptr<model::AbstractPhysicEngine> &engine) override {
        reset_drawables_call_count++;
    }
};

class EnvironmentTest : public EngineTestFixture {};

#endif// ARENAI_CORE_TESTS_ENVIRONMENT_H
