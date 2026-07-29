//
// Created by samuel on 30/06/2026.
//

#include <filesystem>
#include <fstream>

#include <metrics/mean_metric.h>
#include <metrics/metric_saver.h>

#include <arenai_agent_tests/tests_metrics/tests_metrics.h>

using namespace arenai;
using namespace arenai::agent;

namespace {
    std::filesystem::path make_temp_dir() {
        const auto dir =
            std::filesystem::temp_directory_path() / ("arenai_test_" + std::to_string(std::rand()));
        std::filesystem::create_directories(dir);
        return dir;
    }

    int count_lines(const std::filesystem::path &file) {
        std::ifstream f(file);
        int count = 0;
        std::string line;
        while (std::getline(f, line)) count++;
        return count;
    }

    std::string read_first_line(const std::filesystem::path &file) {
        std::ifstream f(file);
        std::string line;
        std::getline(f, line);
        return line;
    }
}// namespace

// ========================================================================
// Fixed tests
// ========================================================================

TEST_F(MetricCsvSaverTest, CreatesFileWithHeader) {
    const auto dir = make_temp_dir();

    auto m1 = std::make_shared<MeanMetric>("loss", 5);
    auto m2 = std::make_shared<MeanMetric>("reward", 5);

    MetricCsvSaver saver(dir, {m1, m2}, 1);

    const auto csv_path = dir / "metrics.csv";
    ASSERT_TRUE(std::filesystem::exists(csv_path));

    const auto header = read_first_line(csv_path);
    ASSERT_TRUE(header.find("loss") != std::string::npos);
    ASSERT_TRUE(header.find("reward") != std::string::npos);
    ASSERT_TRUE(header.find("index") != std::string::npos);

    std::filesystem::remove_all(dir);
}

TEST_F(MetricCsvSaverTest, CreatesParentFolderIfMissing) {
    const auto dir = std::filesystem::temp_directory_path()
                     / ("arenai_test_nested_" + std::to_string(std::rand())) / "sub" / "folder";

    auto m = std::make_shared<MeanMetric>("x", 5);

    MetricCsvSaver saver(dir, {m}, 1);

    ASSERT_TRUE(std::filesystem::exists(dir / "metrics.csv"));

    std::filesystem::remove_all(dir.parent_path().parent_path());
}

TEST_F(MetricCsvSaverTest, AppendsDataRows) {
    const auto dir = make_temp_dir();

    auto m = std::make_shared<MeanMetric>("val", 5);
    m->add(1.0f);

    MetricCsvSaver saver(dir, {m}, 1);

    saver.attempt_append_to_csv();
    saver.attempt_append_to_csv();
    saver.attempt_append_to_csv();

    // 1 header + 3 data rows
    ASSERT_EQ(count_lines(dir / "metrics.csv"), 4);

    std::filesystem::remove_all(dir);
}

// ========================================================================
// Parameterized: save_every behavior
// ========================================================================

TEST_P(MetricCsvSaverParamTest, SaveEveryRespected) {
    const auto [window_size, save_every, total_calls] = GetParam();

    const auto dir = make_temp_dir();

    auto m = std::make_shared<MeanMetric>("val", window_size);
    m->add(1.0f);

    MetricCsvSaver saver(dir, {m}, save_every);

    for (int i = 0; i < total_calls; i++) saver.attempt_append_to_csv();

    // index goes from 0..total_calls-1
    // saves when index % save_every == 0, i.e. at indices 0, save_every, 2*save_every, ...
    const int expected_data_rows = (total_calls + save_every - 1) / save_every;
    const int total_lines = 1 + expected_data_rows;// header + data

    ASSERT_EQ(count_lines(dir / "metrics.csv"), total_lines);

    std::filesystem::remove_all(dir);
}

INSTANTIATE_TEST_SUITE_P(
    MetricCsvSaver, MetricCsvSaverParamTest,
    testing::Combine(
        testing::Values(5, 10), testing::Values(1, 2, 5, 10), testing::Values(1, 5, 10, 20, 50)));
