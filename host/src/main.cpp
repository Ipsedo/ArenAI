//
// Created by samuel on 28/09/2025.
//
#include <iostream>

#include <phyvr_core/environment.h>
#include <phyvr_model/engine.h>
#include <phyvr_view/framebuffer_renderer.h>

#include "./networks/agent.h"
#include "./train.h"
#include "./utils/saver.h"

int main(int argc, char **argv) {
    const auto actor = SacActor(ENEMY_PROPRIOCEPTION_SIZE, ENEMY_NB_ACTION, 256, 512);
    auto critic = std::make_shared<SacCritic>(ENEMY_PROPRIOCEPTION_SIZE, ENEMY_NB_ACTION, 256, 512);

    const auto output_dir = "/home/samuel/Téléchargements/actor_export";

    const std::filesystem::path path(output_dir);

    export_state_dict_neutral(static_cast<torch::nn::Module>(actor), output_dir);

    return 0;
}
