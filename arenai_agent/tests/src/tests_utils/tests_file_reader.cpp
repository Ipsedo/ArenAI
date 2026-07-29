//
// Created by samuel on 30/06/2026.
//

#include <cstdio>
#include <fstream>

#include <SOIL2.h>

#include <arenai_agent/file_reader.h>
#include <arenai_agent_tests/tests_utils/tests_file_reader.h>

using namespace arenai;
using namespace arenai::agent;

void DesktopAssetFileReaderTest::SetUp() {
    tmp_dir = std::filesystem::temp_directory_path() / "arenai_test_file_reader";
    std::filesystem::create_directories(tmp_dir);
}

void DesktopAssetFileReaderTest::TearDown() { std::filesystem::remove_all(tmp_dir); }

// ========================================================================
// read_text tests
// ========================================================================

TEST_F(DesktopAssetFileReaderTest, ReadTextReturnsContent) {
    std::ofstream(tmp_dir / "hello.txt") << "Hello World!";

    DesktopAssetFileReader reader(tmp_dir);
    const auto content = reader.read_text("hello.txt");

    ASSERT_EQ(content, "Hello World!");
}

TEST_F(DesktopAssetFileReaderTest, ReadTextEmptyFile) {
    std::ofstream(tmp_dir / "empty.txt");

    DesktopAssetFileReader reader(tmp_dir);
    const auto content = reader.read_text("empty.txt");

    ASSERT_TRUE(content.empty());
}

TEST_F(DesktopAssetFileReaderTest, ReadTextMultiLine) {
    // written in binary so the file holds exactly these bytes on every
    // platform: read_text is byte-exact (it also serves binary assets)
    std::ofstream(tmp_dir / "multi.txt", std::ios::binary) << "line1\nline2\nline3";

    DesktopAssetFileReader reader(tmp_dir);
    const auto content = reader.read_text("multi.txt");

    ASSERT_EQ(content, "line1\nline2\nline3");
}

TEST_F(DesktopAssetFileReaderTest, ReadTextSubdirectory) {
    std::filesystem::create_directories(tmp_dir / "sub");
    std::ofstream(tmp_dir / "sub" / "data.txt") << "nested";

    DesktopAssetFileReader reader(tmp_dir);
    const auto content = reader.read_text("sub/data.txt");

    ASSERT_EQ(content, "nested");
}

TEST_F(DesktopAssetFileReaderTest, ReadTextThrowsOnMissing) {
    DesktopAssetFileReader reader(tmp_dir);

    ASSERT_THROW(reader.read_text("nonexistent.txt"), std::runtime_error);
}

TEST_F(DesktopAssetFileReaderTest, ReadTextUTF8Content) {
    std::ofstream(tmp_dir / "utf8.txt") << "café résumé naïve";

    DesktopAssetFileReader reader(tmp_dir);
    const auto content = reader.read_text("utf8.txt");

    ASSERT_EQ(content, "café résumé naïve");
}

// ========================================================================
// read_png tests
// ========================================================================

TEST_F(DesktopAssetFileReaderTest, ReadPngValidImage) {
    constexpr int W = 4, H = 3, C = 4;
    std::vector<uint8_t> pixels(W * H * C);
    for (int i = 0; i < W * H * C; ++i) pixels[i] = static_cast<uint8_t>(i % 256);

    const auto png_path = (tmp_dir / "test.png").string();
    ASSERT_TRUE(SOIL_save_image(png_path.c_str(), SOIL_SAVE_TYPE_PNG, W, H, C, pixels.data()))
        << "Failed to create test PNG";

    DesktopAssetFileReader reader(tmp_dir);
    const auto img = reader.read_png("test.png");

    ASSERT_EQ(img.width, W);
    ASSERT_EQ(img.height, H);
    ASSERT_EQ(img.channels, 4);
    ASSERT_EQ(img.pixels.size(), static_cast<size_t>(W * H * 4));
}

TEST_F(DesktopAssetFileReaderTest, ReadPngPixelsNotAllZero) {
    constexpr int W = 2, H = 2, C = 4;
    std::vector<uint8_t> pixels(W * H * C, 200);

    const auto png_path = (tmp_dir / "bright.png").string();
    SOIL_save_image(png_path.c_str(), SOIL_SAVE_TYPE_PNG, W, H, C, pixels.data());

    DesktopAssetFileReader reader(tmp_dir);
    const auto img = reader.read_png("bright.png");

    bool any_non_zero = false;
    for (const auto p: img.pixels)
        if (p > 0) {
            any_non_zero = true;
            break;
        }

    ASSERT_TRUE(any_non_zero);
}

TEST_F(DesktopAssetFileReaderTest, ReadPngThrowsOnMissing) {
    DesktopAssetFileReader reader(tmp_dir);

    ASSERT_THROW(reader.read_png("nonexistent.png"), std::runtime_error);
}

TEST_F(DesktopAssetFileReaderTest, ReadPngAlwaysRGBA) {
    constexpr int W = 2, H = 2, C = 3;
    std::vector<uint8_t> pixels(W * H * C, 128);

    const auto png_path = (tmp_dir / "rgb.png").string();
    SOIL_save_image(png_path.c_str(), SOIL_SAVE_TYPE_PNG, W, H, C, pixels.data());

    DesktopAssetFileReader reader(tmp_dir);
    const auto img = reader.read_png("rgb.png");

    ASSERT_EQ(img.channels, 4);
    ASSERT_EQ(img.pixels.size(), static_cast<size_t>(W * H * 4));
}

TEST_F(DesktopAssetFileReaderTest, ReadPngLargerImage) {
    constexpr int W = 64, H = 32, C = 4;
    std::vector<uint8_t> pixels(W * H * C);
    for (int i = 0; i < W * H * C; ++i) pixels[i] = static_cast<uint8_t>(i % 256);

    const auto png_path = (tmp_dir / "large.png").string();
    SOIL_save_image(png_path.c_str(), SOIL_SAVE_TYPE_PNG, W, H, C, pixels.data());

    DesktopAssetFileReader reader(tmp_dir);
    const auto img = reader.read_png("large.png");

    ASSERT_EQ(img.width, W);
    ASSERT_EQ(img.height, H);
    ASSERT_EQ(img.pixels.size(), static_cast<size_t>(W * H * 4));
}
