//
// Created by samuel on 11/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_METRIC_SAVER_H
#define ARENAI_TRAIN_HOST_METRIC_SAVER_H

#include <filesystem>

#include "./metric.h"

namespace arenai::train {

    class MetricCsvSaver {
    public:
        MetricCsvSaver(
            const std::filesystem::path &output_folder,
            const std::vector<std::shared_ptr<AbstractMetric>> &metrics, int save_every);

        void attempt_append_to_csv();

    private:
        std::filesystem::path csv_file_path;
        std::vector<std::shared_ptr<AbstractMetric>> metrics;

        std::string sep;

        int save_every;
        long index;
    };

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_METRIC_SAVER_H
