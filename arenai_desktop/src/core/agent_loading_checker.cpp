//
// Created by samuel on 19/07/2026.
//

#include "./agent_loading_checker.h"

#include <arenai_agent/factory_set.h>
#include <arenai_core/constants.h>
#include <arenai_utils/exceptions.h>

namespace arenai::desktop {

    std::optional<std::string>
    check_agent_folder(const ModelOptions &model_options, const std::filesystem::path &folder) {
        try {

            const auto agent =
                agent::SacAgentFactory(model_options.hyper_parameters)
                    .get_agent(
                        model_options.vision_height, model_options.vision_width,
                        model::ENEMY_PROPRIOCEPTION_SIZE, model::ENEMY_NB_CONTINUOUS_ACTION,
                        model::ENEMY_NB_DISCRETE_ACTION);

            // stays on CPU: the point is only to prove the state dicts load
            agent->load(folder);

            return std::nullopt;
        } catch (utils::FileDoesNotExistException &e) {
            return "Missing file: " + e.missing_file().filename().string();
        } catch (utils::ModelLoadException &e) {
            return "Error while loading file: " + e.wrong_state_dict_file().filename().string();
        } catch (const std::exception &e) { return "Unknow error while loading Model"; }
    }

}// namespace arenai::desktop
