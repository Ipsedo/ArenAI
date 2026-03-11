//
// Created by samuel on 11/03/2026.
//

#include <arenai_core/constants.h>
#include <arenai_train/factory_set.h>

int main(int argc, char **argv) {

    const auto sac_agent = SacAgentFactory({}).get_agent(
        ENEMY_VISION_HEIGHT, ENEMY_VISION_WIDTH, ENEMY_PROPRIOCEPTION_SIZE,
        ENEMY_NB_CONTINUOUS_ACTION, ENEMY_NB_DISCRETE_ACTION);

    sac_agent->load(std::filesystem::path(
        "/home/samuel/StudioProjects/PhyVR/arenai/arenai_train/outputs/train_131/save_0"));

    std::cout << "Parameters : " << sac_agent->count_parameters() << std::endl;

    return 0;
}
