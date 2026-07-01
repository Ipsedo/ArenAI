//
// Created by samuel on 01/07/2026.
//

#include <filesystem>

#include <arenai_core_tests/utils/engine_test_fixture.h>
#include <arenai_core_tests/utils/local_file_reader.h>
#include <arenai_core_tests/utils/local_gl_context.h>

#ifndef ARENAI_ASSETS_DIR
#error "ARENAI_ASSETS_DIR must be defined"
#endif

void EngineTestFixture::SetUp() {
    constexpr float frequency = 1.f / 60.f;
    engine = make_physic_engine(frequency);

    file_reader = std::make_shared<LocalAssetFileReader>(std::filesystem::path(ARENAI_ASSETS_DIR));

    gl_context = std::make_shared<LocalGlContext>();

    tank_factory = make_tank_factory(*engine, file_reader, frequency);
}
