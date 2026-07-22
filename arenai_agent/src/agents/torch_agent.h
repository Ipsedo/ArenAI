//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_TORCH_AGENT_H
#define ARENAI_TORCH_AGENT_H

#include "./torch_types.h"

namespace arenai::agent {

    class AbstractTorchAgent {
    public:
        virtual ~AbstractTorchAgent() = default;

        virtual TorchAction act(const TorchState &state) = 0;
    };

}// namespace arenai::agent

#endif//ARENAI_TORCH_AGENT_H
