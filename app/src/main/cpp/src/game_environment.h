//
// Created by samuel on 21/03/2023.
//

#ifndef PHYVR_GAME_ENVIRONMENT_H
#define PHYVR_GAME_ENVIRONMENT_H

#include <random>

#include <android_native_app_glue.h>

#include <phyvr_controller/controller.h>
#include <phyvr_core/environment.h>
#include <phyvr_model/engine.h>
#include <phyvr_model/tank_factory.h>
#include <phyvr_utils/file_reader.h>
#include <phyvr_view/renderer.h>

#include "./android_gl_context.h"
#include "./controller/controller_engine.h"

class UserGameTanksEnvironment : public BaseTanksEnvironment {
public:
  explicit UserGameTanksEnvironment(struct android_app *app, int nb_tanks, int threads_num);

  bool is_running() const;

  int32_t on_input(struct android_app *new_app, AInputEvent *event);

  void on_cmd(struct android_app *app, int32_t cmd);

  void pause();

    static void reset_singleton();

protected:
  void on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override;

  void on_reset_physics(const std::unique_ptr<PhysicEngine> &engine) override;

  void on_reset_drawables(
    const std::unique_ptr<PhysicEngine> &engine,
    const std::shared_ptr<AbstractGLContext> &gl_context) override;

private:
  android_app *app;

  std::unique_ptr<TankFactory> tank_factory;

  std::unique_ptr<PlayerRenderer> player_renderer;
  std::unique_ptr<ControllerEngine> player_controller_engine;

  bool is_paused;
};

#endif// PHYVR_GAME_ENVIRONMENT_H
