//
// Created by samuel on 11/03/2026.
//

#ifndef ARENAI_TRAIN_HOST_FACTORY_SET_H
#define ARENAI_TRAIN_HOST_FACTORY_SET_H

#include "./factory.h"

namespace arenai::train {

    class SacAgentFactory : public AgentFactory {
    public:
        explicit SacAgentFactory(const std::map<std::string, std::string> &arguments);

    protected:
        std::shared_ptr<AbstractAgent> get_agent_impl(
            const int &vision_height, const int &vision_width, const int &nb_sensors,
            const int &nb_continuous_actions, const int &nb_discrete_action) override;
    };

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_FACTORY_SET_H
