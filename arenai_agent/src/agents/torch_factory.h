//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_TORCH_FACTORY_H
#define ARENAI_TORCH_FACTORY_H

#include <map>
#include <memory>
#include <string>

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

        // the algorithm's resolved hyper-parameters, keyed by CLI option name:
        // dumped next to the metrics so a run stays identifiable afterwards
        virtual std::map<std::string, std::string> get_config() const = 0;
    };

}// namespace arenai::agent

#endif//ARENAI_TORCH_FACTORY_H
