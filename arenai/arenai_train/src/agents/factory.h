//
// Created by samuel on 22/01/2026.
//

#ifndef ARENAI_TRAIN_HOST_FACTORY_H
#define ARENAI_TRAIN_HOST_FACTORY_H

#include <map>
#include <memory>
#include <string>

#include "./agent.h"

class AgentFactory {
public:
    virtual ~AgentFactory() = default;

    explicit AgentFactory(const std::map<std::string, std::string> &arguments);

    std::shared_ptr<AbstractAgent> get_agent(
        const int &vision_height, const int &vision_width, const int &nb_sensors,
        const int &nb_actions);

protected:
    template<typename T>
    T get_value(const std::string &argument_name, T default_value) {
        if (!arguments.contains(argument_name)) return default_value;

        const std::string value_as_string = arguments[argument_name];
        std::stringstream ss(value_as_string);
        T value;
        ss >> value;

        if (ss.fail() || !ss.eof())
            throw std::runtime_error(std::format(
                "Wrong value for \"{}\" : \"{}\", example : \"{}\"", argument_name, value_as_string,
                default_value));

        arguments.erase(arguments.find(argument_name));

        return value;
    }

    virtual std::shared_ptr<AbstractAgent> get_agent_impl(
        const int &vision_height, const int &vision_width, const int &nb_sensors,
        const int &nb_actions) = 0;

private:
    std::map<std::string, std::string> arguments;
};

class CnnAgentFactory : public AgentFactory {
protected:
    std::shared_ptr<AbstractAgent> get_agent_impl(
        const int &vision_height, const int &vision_width, const int &nb_sensors,
        const int &nb_actions) override;

    virtual std::shared_ptr<AbstractAgent> get_cnn_agent_impl(
        const int &vision_height, const int &vision_width, const int &nb_sensors,
        const int &nb_actions, const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums) = 0;
};

#endif//ARENAI_TRAIN_HOST_FACTORY_H
