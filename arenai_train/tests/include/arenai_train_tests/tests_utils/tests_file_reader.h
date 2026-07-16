//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_FILE_READER_H
#define ARENAI_TESTS_FILE_READER_H

#include <filesystem>

#include <gtest/gtest.h>

class DesktopAssetFileReaderTest : public testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;

    std::filesystem::path tmp_dir;
};

#endif//ARENAI_TESTS_FILE_READER_H
