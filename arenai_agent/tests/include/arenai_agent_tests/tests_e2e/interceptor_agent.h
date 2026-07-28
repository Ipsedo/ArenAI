//
// Created by samuel on 28/07/2026.
//

#ifndef ARENAI_AGENT_TESTS_INTERCEPTOR_AGENT_H
#define ARENAI_AGENT_TESTS_INTERCEPTOR_AGENT_H

#include <vector>

#include <arenai_agent/agent.h>

// Records every batch of states the host loop hands to the agent and answers
// with neutral actions: what act() received is exactly what a real network
// would have seen.
class InterceptorAgent final : public arenai::agent::AbstractAgent {
public:
    std::vector<std::vector<arenai::core::State>> received_states;
    int last_vision_height = -1;
    int last_vision_width = -1;

    std::vector<arenai::core::Action>
    act(const std::vector<arenai::core::State> &states, const int vision_height,
        const int vision_width) override {
        received_states.push_back(states);
        last_vision_height = vision_height;
        last_vision_width = vision_width;

        return std::vector<arenai::core::Action>(states.size(), {{0.f, 0.f}, {0.f, 0.f}, {false}});
    }

    void load(const std::filesystem::path &agent_folder) override {}
};

#endif// ARENAI_AGENT_TESTS_INTERCEPTOR_AGENT_H
