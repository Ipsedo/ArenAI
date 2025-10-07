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
    auto actor = SacActor(ENEMY_PROPRIOCEPTION_SIZE, ENEMY_NB_ACTION, 64, 256);
    auto critic = SacCritic(ENEMY_PROPRIOCEPTION_SIZE, ENEMY_NB_ACTION, 64, 256);

    const auto v = torch::randn({2, 3, ENEMY_VISION_SIZE, ENEMY_VISION_SIZE});
    const auto p = torch::randn({2, ENEMY_PROPRIOCEPTION_SIZE});

    const auto [mu, sigma] = actor.act(v, p);
    const auto value = critic.value(v, p, mu);

    std::cout << "mu :" << std::endl << mu << std::endl;
    std::cout << "sigma :" << std::endl << sigma << std::endl;
    std::cout << "value :" << std::endl << value << std::endl;

    const auto output_dir = "/home/samuel/Téléchargements/actor_export";

    const std::filesystem::path path(output_dir);

    export_state_dict_neutral(static_cast<torch::nn::Module>(actor), output_dir);

    std::cout << "model saved" << std::endl;

    return 0;
}
