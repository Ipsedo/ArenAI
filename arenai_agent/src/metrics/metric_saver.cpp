//
// Created by samuel on 11/06/2026.
//

#include "./metric_saver.h"

#include <filesystem>
#include <fstream>

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    MetricCsvSaver::MetricCsvSaver(
        const std::filesystem::path &output_folder,
        const std::vector<std::shared_ptr<AbstractMetric>> &metrics, const int save_every)
        : csv_file_path(output_folder / "metrics.csv"), metrics(metrics), sep(";"),
          save_every(save_every), index(0L) {

        if (!std::filesystem::exists(csv_file_path.parent_path()))
            std::filesystem::create_directories(csv_file_path.parent_path());

        std::ofstream file(csv_file_path, std::ios::out | std::ios::trunc);

        for (const auto &m: metrics) {
            file << m->get_name();
            file << sep;
        }

        file << "index\n";

        file.close();
    }

    void MetricCsvSaver::attempt_append_to_csv() {
        if (index % save_every == 0) {
            std::ofstream file(csv_file_path, std::ios::out | std::ios::app);

            for (const auto &m: metrics) {
                file << std::to_string(m->compute_metric());
                file << sep;
            }

            file << std::to_string(index) + "\n";

            file.close();
        }

        index++;
    }

}// namespace arenai::agent
