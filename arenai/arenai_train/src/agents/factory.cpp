//
// Created by samuel on 22/01/2026.
//

#include "./factory.h"

#include <iostream>

AgentFactory::AgentFactory(const std::map<std::string, std::string> &arguments)
    : arguments(arguments) {}

std::shared_ptr<AbstractAgent> AgentFactory::get_agent(
    const int &vision_height, const int &vision_width, const int &nb_sensors,
    const int &nb_actions) {
    const auto agent = get_agent_impl(vision_height, vision_width, nb_sensors, nb_actions);

    if (!arguments.empty()) {
        std::cerr << "Invalid argument(s) : " << arguments << std::endl;
        throw std::runtime_error("Invalid argument(s)");
    }

    return agent;
}

std::shared_ptr<AbstractAgent> CnnAgentFactory::get_agent_impl(
    const int &vision_height, const int &vision_width, const int &nb_sensors,
    const int &nb_actions) {}
