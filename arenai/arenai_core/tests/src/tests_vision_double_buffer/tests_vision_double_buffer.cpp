//
// Created by samuel on 01/07/2026.
//

#include <arenai_core/thread_pool.h>
#include <arenai_core_tests/tests_vision_double_buffer/tests_vision_double_buffer.h>

// ========================================================================
// Construction — initial black image
// ========================================================================

TEST_F(VisionDoubleBufferTest, InitialImageIsBlack) {
    constexpr int height = 16;
    constexpr int width = 16;

    const VisionDoubleBuffer buffer(height, width);

    const auto [pixels] = buffer.read_copy();

    ASSERT_EQ(static_cast<int>(pixels.size()), 3 * height * width);

    for (const auto pixel: pixels) { ASSERT_EQ(pixel, 0); }
}

TEST_F(VisionDoubleBufferTest, InitialImageSizeMatchesDimensions) {
    constexpr int height = 32;
    constexpr int width = 64;

    const VisionDoubleBuffer buffer(height, width);

    const auto [pixels] = buffer.read_copy();

    ASSERT_EQ(static_cast<int>(pixels.size()), 3 * height * width);
}

// ========================================================================
// Write / Read round-trip
// ========================================================================

TEST_F(VisionDoubleBufferTest, WriteAndReadBack) {
    constexpr int height = 4;
    constexpr int width = 4;
    constexpr int total = 3 * height * width;

    VisionDoubleBuffer buffer(height, width);

    image<uint8_t> written{};
    written.pixels.resize(total);
    for (int i = 0; i < total; i++) written.pixels[i] = static_cast<uint8_t>(i % 256);

    buffer.write(written);

    const auto [pixels] = buffer.read_copy();

    ASSERT_EQ(pixels.size(), written.pixels.size());
    for (int i = 0; i < total; i++) {
        ASSERT_EQ(pixels[i], written.pixels[i]) << "mismatch at index " << i;
    }
}

TEST_F(VisionDoubleBufferTest, SecondWriteOverwritesFirst) {
    constexpr int height = 2;
    constexpr int width = 2;
    constexpr int total = 3 * height * width;

    VisionDoubleBuffer buffer(height, width);

    image<uint8_t> first{std::vector<uint8_t>(total, 100)};
    buffer.write(first);

    image<uint8_t> second{std::vector<uint8_t>(total, 200)};
    buffer.write(second);

    const auto [pixels] = buffer.read_copy();

    for (int i = 0; i < total; i++) { ASSERT_EQ(pixels[i], 200); }
}

// ========================================================================
// Square and non-square sizes
// ========================================================================

TEST_F(VisionDoubleBufferTest, NonSquareDimensions) {
    constexpr int height = 8;
    constexpr int width = 32;

    const VisionDoubleBuffer buffer(height, width);

    const auto [pixels] = buffer.read_copy();

    ASSERT_EQ(static_cast<int>(pixels.size()), 3 * height * width);
}
