//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_TRAIN_ENVIRONMENT_H
#define ARENAI_TRAIN_HOST_TRAIN_ENVIRONMENT_H

#include <arenai_core/environment.h>

class TrainTankEnvironment final : public BaseTanksEnvironment {
public:
    TrainTankEnvironment(
        int nb_tanks, const std::filesystem::path &android_assets_path, float wanted_frequency);

    std::vector<std::tuple<State, Reward, IsDone>>
    step(float time_delta, std::future<std::vector<Action>> &actions_future) override;

    std::vector<Reward> get_potential_rewards();

    static void reset_singleton();

protected:
    void on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override;

    void on_reset_physics(const std::unique_ptr<PhysicEngine> &engine) override;

    void on_reset_drawables(
        const std::unique_ptr<PhysicEngine> &engine,
        const std::shared_ptr<AbstractGLContext> &gl_context) override;

private:
    int max_frames_without_shoot;
    std::vector<int> remaining_frames;
    int nb_frames_added_when_shoot;
    int nb_tanks;
};

#endif// ARENAI_TRAIN_HOST_TRAIN_ENVIRONMENT_H
