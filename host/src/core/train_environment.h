//
// Created by samuel on 03/10/2025.
//

#ifndef PHYVR_TRAIN_HOST_TRAIN_ENVIRONMENT_H
#define PHYVR_TRAIN_HOST_TRAIN_ENVIRONMENT_H

#include <phyvr_core/environment.h>

class TrainTankEnvironment final : public BaseTanksEnvironment {
public:
    explicit TrainTankEnvironment(int nb_tanks, const std::filesystem::path &android_assets_path);

    static void reset_singleton();

protected:
    void on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override;

    void on_reset_physics(const std::unique_ptr<PhysicEngine> &engine) override;

    void on_reset_drawables(
        const std::unique_ptr<PhysicEngine> &engine,
        const std::shared_ptr<AbstractGLContext> &gl_context) override;
};

#endif// PHYVR_TRAIN_HOST_TRAIN_ENVIRONMENT_H
