//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_TORCH_FACTORY_H
#define ARENAI_TORCH_FACTORY_H

#include <memory>

#include "./step_collector.h"
#include "./torch_agent.h"
#include "./trainer.h"

namespace arenai::agent {

    // Builds and wires the triad of one algorithm: the three views share the
    // same internal state (actor, buffer) - the factory guarantees consistency.
    class AbstractTorchAgentFactory {
    public:
        virtual ~AbstractTorchAgentFactory() = default;

        virtual std::shared_ptr<AbstractTorchAgent> get_agent() = 0;
        virtual std::shared_ptr<AbstractStepCollector> get_collector() = 0;
        virtual std::shared_ptr<AbstractTrainer> get_trainer() = 0;
    };

}// namespace arenai::agent

#endif//ARENAI_TORCH_FACTORY_H
