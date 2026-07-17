//
// Created by samuel on 03/10/2025.
//

#include "./torch_saver.h"

#include <fstream>

#include <nlohmann/json.hpp>

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

    /*
     * Agent Saver
     */

    AgentSaver::AgentSaver(
        const std::shared_ptr<AbstractAgent> &agent, const std::filesystem::path &output_path,
        const int save_every)
        : agent(agent), curr_step(0), save_every(save_every), output_path(output_path) {}

    void AgentSaver::attempt_save() {
        if (curr_step % save_every == 0) {
            const auto output_folder =
                output_path / ("save_" + std::to_string(curr_step / save_every));
            if (!std::filesystem::exists(output_folder))
                std::filesystem::create_directories(output_folder);
            agent->save(output_folder);
        }

        curr_step++;
    }

}// namespace arenai::train
