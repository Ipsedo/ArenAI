//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_AGENT_HOST_CLI_FIELDS_H
#define ARENAI_AGENT_HOST_CLI_FIELDS_H

#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <argparse/argparse.hpp>

#include "./cli_parser.h"

namespace arenai::agent {

    // One CLI option bound to a member of the hyper-parameter struct S. The
    // default value is read from a default-constructed S, so the member
    // initializers of S are the single source of truth for CLI defaults.
    template<typename S>
    struct CliField {
        std::string name;
        std::variant<
            int S::*, float S::*, std::vector<int> S::*, std::vector<std::tuple<int, int>> S::*>
            member;
    };

    /*
     * Add field
     */

    inline void add_cli_field(
        argparse::ArgumentParser &parser, const std::string &name, const int default_value) {
        parser.add_argument(name).scan<'i', int>().default_value(default_value);
    }

    inline void add_cli_field(
        argparse::ArgumentParser &parser, const std::string &name, const float default_value) {
        parser.add_argument(name).scan<'g', float>().default_value(default_value);
    }

    inline void add_cli_field(
        argparse::ArgumentParser &parser, const std::string &name,
        const std::vector<int> &default_value) {
        parser.add_argument(name)
            .default_value<hidden_layers>({default_value})
            .action([name](const std::string &value) {
                return hidden_layers{parse_int_vector(value, name, "[256, 128, ..., 64]")};
            });
    }

    inline void add_cli_field(
        argparse::ArgumentParser &parser, const std::string &name,
        const std::vector<std::tuple<int, int>> &default_value) {
        parser.add_argument(name)
            .default_value<vision_channels>({default_value})
            .action(parse_cli_vision_channels);
    }

    /*
     * Read fields
     */

    inline void
    read_cli_field(const argparse::ArgumentParser &parser, const std::string &name, int &output) {
        output = parser.get<int>(name);
    }

    inline void
    read_cli_field(const argparse::ArgumentParser &parser, const std::string &name, float &output) {
        output = parser.get<float>(name);
    }

    inline void read_cli_field(
        const argparse::ArgumentParser &parser, const std::string &name, std::vector<int> &output) {
        output = parser.get<hidden_layers>(name).layers;
    }

    inline void read_cli_field(
        const argparse::ArgumentParser &parser, const std::string &name,
        std::vector<std::tuple<int, int>> &output) {
        output = parser.get<vision_channels>(name).channels;
    }

    /*
     * Add & read cli arg
     */

    template<typename S>
    void add_cli_fields(argparse::ArgumentParser &parser, const std::vector<CliField<S>> &fields) {
        const S default_params{};
        for (const auto &field: fields)
            std::visit(
                [&](const auto member) {
                    add_cli_field(parser, field.name, default_params.*member);
                },
                field.member);
    }

    template<typename S>
    S read_cli_fields(
        const argparse::ArgumentParser &parser, const std::vector<CliField<S>> &fields) {
        S params{};
        for (const auto &field: fields)
            std::visit(
                [&](const auto member) { read_cli_field(parser, field.name, params.*member); },
                field.member);
        return params;
    }

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_CLI_FIELDS_H
