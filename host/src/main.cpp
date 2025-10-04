//
// Created by samuel on 28/09/2025.
//
#include <iostream>

#include "./networks/agent.h"
#include "./train.h"
#include "./utils/saver.h"
#include <phyvr_model/engine.h>
#include <phyvr_view/framebuffer_renderer.h>

int main(int argc, char **argv) {
  std::cout << "toyo" << std::endl;

  constexpr int nb_sensors = 10;
  constexpr int nb_actions = 8;
  const auto actor = SacActor(nb_sensors, nb_actions, 128, 256);
  auto critic = std::make_shared<SacCritic>(nb_sensors, nb_actions, 128, 256);

  const auto output_dir = "/home/samuel/Téléchargements/actor_export";

  const std::filesystem::path path(output_dir);

  export_state_dict_neutral(static_cast<torch::nn::Module>(actor), output_dir);

  return 0;
}
