//
// Created by samuel on 01/07/2026.
//

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <arenai_core/thread_pool.h>
#include <arenai_core_tests/tests_model_matrices_double_buffer/tests_model_matrices_double_buffer.h>

using namespace arenai;
using namespace arenai::core;

// ========================================================================
// Construction — initial empty vector
// ========================================================================

TEST_F(ModelMatricesDoubleBufferTest, InitialReadIsEmpty) {
    const ModelMatricesDoubleBuffer buffer;

    const auto matrices = buffer.read_copy();

    ASSERT_TRUE(matrices.empty());
}

// ========================================================================
// Write / Read round-trip
// ========================================================================

TEST_F(ModelMatricesDoubleBufferTest, WriteAndReadBack) {
    ModelMatricesDoubleBuffer buffer;

    constexpr glm::mat4 identity(1.f);
    const glm::mat4 scaled = glm::scale(glm::mat4(1.f), glm::vec3(2.f));

    std::vector<std::tuple<std::string, glm::mat4>> written;
    written.emplace_back("item_a", identity);
    written.emplace_back("item_b", scaled);

    buffer.write(written);

    const auto read = buffer.read_copy();

    ASSERT_EQ(read.size(), 2);
    ASSERT_EQ(std::get<0>(read[0]), "item_a");
    ASSERT_EQ(std::get<0>(read[1]), "item_b");

    ASSERT_EQ(std::get<1>(read[0]), identity);
    ASSERT_EQ(std::get<1>(read[1]), scaled);
}

TEST_F(ModelMatricesDoubleBufferTest, SecondWriteOverwritesFirst) {
    ModelMatricesDoubleBuffer buffer;

    std::vector<std::tuple<std::string, glm::mat4>> first;
    first.emplace_back("old", glm::mat4(1.f));
    buffer.write(first);

    std::vector<std::tuple<std::string, glm::mat4>> second;
    second.emplace_back("new_a", glm::mat4(1.f));
    second.emplace_back("new_b", glm::mat4(1.f));
    second.emplace_back("new_c", glm::mat4(1.f));
    buffer.write(second);

    const auto read = buffer.read_copy();

    ASSERT_EQ(read.size(), 3);
    ASSERT_EQ(std::get<0>(read[0]), "new_a");
    ASSERT_EQ(std::get<0>(read[1]), "new_b");
    ASSERT_EQ(std::get<0>(read[2]), "new_c");
}

// ========================================================================
// Write empty then non-empty
// ========================================================================

TEST_F(ModelMatricesDoubleBufferTest, WriteEmptyThenNonEmpty) {
    ModelMatricesDoubleBuffer buffer;

    buffer.write(std::vector<std::tuple<std::string, glm::mat4>>{});
    ASSERT_TRUE(buffer.read_copy().empty());

    std::vector<std::tuple<std::string, glm::mat4>> data;
    data.emplace_back("cubemap", glm::scale(glm::mat4(1.f), glm::vec3(2000.f)));
    buffer.write(data);

    const auto read = buffer.read_copy();
    ASSERT_EQ(read.size(), 1);
    ASSERT_EQ(std::get<0>(read[0]), "cubemap");
}
