//
// Created by samuel on 01/07/2026.
//

#ifndef ARENAI_MODEL_TESTS_ENGINE_TEST_FIXTURE_H
#define ARENAI_MODEL_TESTS_ENGINE_TEST_FIXTURE_H

#include <memory>

#include <gtest/gtest.h>

#include <arenai_model/engine.h>
#include <arenai_model/item.h>
#include <arenai_model/tank.h>
#include <arenai_model/tank_factory.h>
#include <arenai_utils/file_reader.h>

class EngineTestFixture : public testing::Test {
protected:
    void SetUp() override;

    void add_ground();

    std::unique_ptr<arenai::model::AbstractPhysicEngine> engine;
    std::shared_ptr<arenai::utils::AbstractResourceFileReader> file_reader;
    std::shared_ptr<arenai::model::TankFactory> tank_factory;
    std::shared_ptr<arenai::model::Item> ground;
};

#endif// ARENAI_MODEL_TESTS_ENGINE_TEST_FIXTURE_H
