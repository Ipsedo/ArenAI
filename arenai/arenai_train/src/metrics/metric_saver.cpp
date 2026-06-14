//
// Created by samuel on 11/06/2026.
//

#include "./metric_saver.h"

#include <fstream>

MetricCsvSaver::MetricCsvSaver(
    const std::filesystem::path &output_folder,
    const std::vector<std::shared_ptr<AbstractMetric>> &metrics, const int save_every)
    : csv_file_path(output_folder / "metrics.csv"), metrics(metrics), sep(";"),
      save_every(save_every), index(0L) {

    std::ofstream file(csv_file_path, std::ios::out);
    std::string header;

    for (const auto &m: metrics) header += m->get_name() + sep;

    header += "index\n";

    file << header;
}

void MetricCsvSaver::attempt_append_to_csv() {
    if (index % save_every == 0) {
        std::ofstream file(csv_file_path, std::ios::app);

        for (const auto &m: metrics) file << std::to_string(m->compute_metric()) + sep;

        file << std::to_string(index) + "\n";
    }

    index++;
}
