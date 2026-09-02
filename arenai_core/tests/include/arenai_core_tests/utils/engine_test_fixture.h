//
// Created by samuel on 01/07/2026.
//

#ifndef ARENAI_CORE_TESTS_ENGINE_TEST_FIXTURE_H
#define ARENAI_CORE_TESTS_ENGINE_TEST_FIXTURE_H

#include <memory>

#include <gtest/gtest.h>

#include <arenai_model/engine.h>
#include <arenai_model/item.h>
#include <arenai_model/tank.h>
#include <arenai_model/tank_factory.h>
#include <arenai_utils/file_reader.h>
#include <arenai_view/backend.h>

class EngineTestFixture : public testing::Test {
protected:
    void SetUp() override;

    std::unique_ptr<model::AbstractPhysicEngine> engine;
    std::shared_ptr<utils::AbstractResourceFileReader> file_reader;
    std::shared_ptr<view::AbstractGraphicBackend> graphics_backend;
    std::shared_ptr<model::TankFactory> tank_factory;
};

#endif// ARENAI_CORE_TESTS_ENGINE_TEST_FIXTURE_H
