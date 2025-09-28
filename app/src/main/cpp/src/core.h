//
// Created by samuel on 21/03/2023.
//

#ifndef PHYVR_CORE_H
#define PHYVR_CORE_H

#include <android_native_app_glue.h>
#include <random>

#include "./controller/controller_engine.h"
#include <phyvr_controller/controller.h>
#include <phyvr_model/engine.h>
#include <phyvr_model/tank_factory.h>
#include <phyvr_utils/file_reader.h>
#include <phyvr_view/renderer.h>

class CoreEngine {
public:
  explicit CoreEngine(struct android_app *app);

  bool is_running() const;

  void draw();

  void step(float time_delta);

  int32_t on_input(struct android_app *app, AInputEvent *event);

  void on_cmd(struct android_app *app, int32_t cmd);

private:
  std::shared_ptr<AbstractFileReader> file_reader;

  TankFactory tank_factory;

  std::shared_ptr<Camera> camera;

  std::unique_ptr<PhysicEngine> physic_engine;
  std::unique_ptr<Renderer> renderer;
  std::unique_ptr<ControllerEngine> controller_engine;

  std::random_device dev;
  std::mt19937 rng;

  bool is_paused;

  void _new_view(struct android_app *app);

  void _pause();
};

#endif // PHYVR_CORE_H
