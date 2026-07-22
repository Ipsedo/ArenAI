//
// Created by samuel on 21/01/2026.
//

#ifndef ARENAI_AGENT_HOST_AGENT_H
#define ARENAI_AGENT_HOST_AGENT_H

#include <filesystem>
#include <vector>

#include <arenai_core/types.h>

namespace arenai::agent {

    class AbstractAgent {
    public:
        virtual ~AbstractAgent() = default;

        virtual std::vector<core::Action>
        act(const std::vector<core::State> &states, int vision_height, int vision_width) = 0;

        virtual void load(const std::filesystem::path &agent_folder) = 0;
    };

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_AGENT_H
