//
// Created by samuel on 01/07/2026.
//

#ifndef ARENAI_CORE_TESTS_ENVIRONMENT_H
#define ARENAI_CORE_TESTS_ENVIRONMENT_H

#include <gtest/gtest.h>

#include <arenai_core/environment.h>

#include "utils/engine_test_fixture.h"

class TestTanksEnvironment final : public arenai::core::BaseTanksEnvironment {
public:
    using arenai::core::BaseTanksEnvironment::BaseTanksEnvironment;

    int draw_call_count = 0;
    int reset_physics_call_count = 0;
    int reset_drawables_call_count = 0;

protected:
    void on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override {
        draw_call_count++;
    }

    void
    on_reset_physics(const std::unique_ptr<arenai::model::AbstractPhysicEngine> &engine) override {
        reset_physics_call_count++;
    }

    void on_reset_drawables(
        const std::unique_ptr<arenai::model::AbstractPhysicEngine> &engine) override {
        reset_drawables_call_count++;
    }
};

class EnvironmentTest : public EngineTestFixture {};

#endif// ARENAI_CORE_TESTS_ENVIRONMENT_H
