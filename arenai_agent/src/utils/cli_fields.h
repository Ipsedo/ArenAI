//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_AGENT_HOST_CLI_FIELDS_H
#define ARENAI_AGENT_HOST_CLI_FIELDS_H

#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>

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

    // argparse's get<T> unpacks container types element-wise, so a
    // vector produced by an action must travel boxed in a scalar type
    template<typename T>
    struct CliJsonValue {
        T value;
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

    template<typename T>
    void add_cli_json_field(
        argparse::ArgumentParser &parser, const std::string &name, const T &default_value) {
        parser.add_argument(name)
            .default_value(CliJsonValue<T>{default_value})
            .action([name](const std::string &value) {
                try {
                    return CliJsonValue<T>{nlohmann::json::parse(value).get<T>()};
                } catch (const nlohmann::json::exception &) {
                    throw std::invalid_argument(
                        "invalid " + name + " value, usage : " + nlohmann::json(T{}).dump()
                        + " (JSON), actual value = \"" + value + "\"");
                }
            });
    }

    inline void add_cli_field(
        argparse::ArgumentParser &parser, const std::string &name,
        const std::vector<int> &default_value) {
        add_cli_json_field(parser, name, default_value);
    }

    inline void add_cli_field(
        argparse::ArgumentParser &parser, const std::string &name,
        const std::vector<std::tuple<int, int>> &default_value) {
        add_cli_json_field(parser, name, default_value);
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
        output = parser.get<CliJsonValue<std::vector<int>>>(name).value;
    }

    inline void read_cli_field(
        const argparse::ArgumentParser &parser, const std::string &name,
        std::vector<std::tuple<int, int>> &output) {
        output = parser.get<CliJsonValue<std::vector<std::tuple<int, int>>>>(name).value;
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

    // the resolved hyper-parameters keyed by option name, dashes stripped: what the
    // run was actually launched with, as native JSON values
    template<typename S>
    nlohmann::json cli_fields_to_json(const std::vector<CliField<S>> &fields, const S &params) {
        nlohmann::json config;
        for (const auto &field: fields)
            std::visit(
                [&](const auto member) {
                    config[field.name.substr(field.name.find_first_not_of('-'))] = params.*member;
                },
                field.member);
        return config;
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
