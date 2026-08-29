//
// Created by claude on 29/08/2026.
//

#ifndef ARENAI_AGENT_HOST_CONFIG_FORMAT_H
#define ARENAI_AGENT_HOST_CONFIG_FORMAT_H

#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace arenai::agent {

    // hyper-parameter values rendered for the run's config dump

    inline std::string format_config_value(const int value) { return std::to_string(value); }

    inline std::string format_config_value(const float value) {
        std::ostringstream stream;
        stream << std::setprecision(6) << value;
        return stream.str();
    }

    inline std::string format_config_value(const std::vector<int> &value) {
        std::ostringstream stream;
        stream << "[";
        for (int i = 0; i < value.size(); i++) stream << (i ? ", " : "") << value[i];
        stream << "]";
        return stream.str();
    }

    inline std::string format_config_value(const std::vector<std::tuple<int, int>> &value) {
        std::ostringstream stream;
        stream << "[";
        for (int i = 0; i < value.size(); i++) {
            const auto &[in_channels, out_channels] = value[i];
            stream << (i ? ", " : "") << "(" << in_channels << ", " << out_channels << ")";
        }
        stream << "]";
        return stream.str();
    }

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_CONFIG_FORMAT_H
