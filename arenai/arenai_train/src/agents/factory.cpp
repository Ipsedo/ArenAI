//
// Created by samuel on 22/01/2026.
//

#include <iostream>

#include <arenai_train/factory.h>

AgentFactory::AgentFactory(const std::map<std::string, std::string> &arguments)
    : arguments(arguments) {}

std::shared_ptr<AbstractAgent> AgentFactory::get_agent(
    const int &vision_height, const int &vision_width, const int &nb_sensors,
    const int &nb_continuous_actions, const int &nb_discrete_actions) {
    const auto agent = get_agent_impl(
        vision_height, vision_width, nb_sensors, nb_continuous_actions, nb_discrete_actions);

    if (!arguments.empty()) {
        std::cerr << "Invalid argument(s) : " << arguments << std::endl;
        throw std::runtime_error("Invalid argument(s)");
    }

    return agent;
}
