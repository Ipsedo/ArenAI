//
// Created by samuel on 01/07/2026.
//

#include <filesystem>

#include <glm/glm.hpp>

#include <arenai_model/item_factory.h>
#include <arenai_model_tests/utils/engine_test_fixture.h>
#include <arenai_model_tests/utils/local_file_reader.h>

using namespace arenai;
using namespace arenai::model;
using namespace arenai::utils;

#ifndef ARENAI_ASSETS_DIR
#error "ARENAI_ASSETS_DIR must be defined"
#endif

void EngineTestFixture::SetUp() {
    constexpr float frequency = 1.f / 60.f;
    engine = make_physic_engine(frequency);

    file_reader = std::make_shared<LocalAssetFileReader>(std::filesystem::path(ARENAI_ASSETS_DIR));

    tank_factory = engine->get_tank_factory();
}

void EngineTestFixture::add_ground() {
    ground = engine->get_item_factory()->make_cube_item(
        "ground", file_reader, glm::vec3(0.f, -0.5f, 0.f), glm::vec3(500.f, 0.5f, 500.f), 0.f);
}
