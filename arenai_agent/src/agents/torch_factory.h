#ifndef ARENAI_TORCH_FACTORY_H
#define ARENAI_TORCH_FACTORY_H

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "./step_collector.h"
#include "./torch_agent.h"
#include "./trainer.h"

namespace arenai::agent {

    class AbstractTorchAgentFactory {
    public:
        virtual ~AbstractTorchAgentFactory() = default;

        virtual std::shared_ptr<AbstractTorchAgent> get_agent() = 0;
        virtual std::shared_ptr<AbstractStepCollector> get_collector() = 0;
        virtual std::shared_ptr<AbstractTrainer> get_trainer() = 0;

        virtual nlohmann::json get_config() const = 0;
    };

}// namespace arenai::agent

#endif//ARENAI_TORCH_FACTORY_H
