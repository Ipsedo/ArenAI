//
// Created by samuel on 21/03/2023.
//

#include "./game_environment.h"

#include <glm/gtx/transform.hpp>

#include <phyvr_model/convex.h>
#include <phyvr_model/engine.h>
#include <phyvr_model/height_map.h>
#include <phyvr_model/shapes.h>
#include <phyvr_model/tank_factory.h>
#include <phyvr_utils/logging.h>
#include <phyvr_view/cubemap.h>
#include <phyvr_view/specular.h>

#include "./android_file_reader.h"
#include "./android_gl_context.h"

UserGameTanksEnvironment::UserGameTanksEnvironment(
  struct android_app *app, int nb_tanks, int threads_num)
    : BaseTanksEnvironment(
      std::make_shared<AndroidFileReader>(app->activity->assetManager),
      std::make_shared<AndroidGLContext>(app->window), nb_tanks, threads_num),
      app(app), tank_factory(std::nullptr_t()), is_paused(true), player_renderer(std::nullptr_t()),
      player_controller_engine(std::nullptr_t()) {}

void UserGameTanksEnvironment::on_draw(
  const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
  player_renderer->draw(model_matrices);
}

int32_t UserGameTanksEnvironment::on_input(struct android_app *app, AInputEvent *event) {
  if (AKeyEvent_getKeyCode(event) == AKEYCODE_BACK) {
    ANativeActivity_finish(app->activity);
    return 1;
  }

  return player_controller_engine->on_event(event);
}

void UserGameTanksEnvironment::on_cmd(struct android_app *new_app, int32_t cmd) {
  switch (cmd) {
    case APP_CMD_SAVE_STATE:
      // The system has asked us to save our current state.  Do so.
      /*app->savedState = malloc(sizeof(UserGameTanksEnvironment));
    // TODO real object copy...
    app->savedState = engine;
    app->savedStateSize = sizeof(UserGameTanksEnvironment);
    */
      LOG_INFO("save state");
      break;
    case APP_CMD_INIT_WINDOW:
      // The window is being shown, get it ready.
      if (app->window != nullptr) {
        LOG_INFO("opening window");
        app = new_app;
        reset_drawables(std::make_shared<AndroidGLContext>(app->window));
        is_paused = false;
      }
      break;
    case APP_CMD_TERM_WINDOW:
      pause();
      LOG_INFO("close");
      break;
    case APP_CMD_GAINED_FOCUS: LOG_INFO("gained focus"); break;
    case APP_CMD_LOST_FOCUS:
      pause();
      LOG_INFO("lost focus");
      break;
    default: break;
  }
}

bool UserGameTanksEnvironment::is_running() const { return !is_paused; }

void UserGameTanksEnvironment::on_reset_physics(const std::shared_ptr<PhysicEngine> &engine) {
  tank_factory = std::make_unique<TankFactory>(file_reader, "player", glm::vec3(0., -40., 40));

  for (auto &item: tank_factory->get_items()) { engine->add_item(item); }

  for (auto &item_producer: tank_factory->get_item_producers())
    engine->add_item_producer(item_producer);
}

void UserGameTanksEnvironment::on_reset_drawables(
  const std::shared_ptr<PhysicEngine> &engine,
  const std::shared_ptr<AbstractGLContext> &gl_context) {
  player_renderer = std::make_unique<PlayerRenderer>(
    gl_context, ANativeWindow_getWidth(app->window), ANativeWindow_getHeight(app->window),
    glm::vec3(200, 300, 200), tank_factory->get_camera());
  player_controller_engine = std::make_unique<ControllerEngine>(
    app->config, player_renderer->get_width(), player_renderer->get_height());

  for (auto &ctrl: tank_factory->get_controllers()) player_controller_engine->add_controller(ctrl);

  player_renderer->add_drawable("cubemap", std::make_unique<CubeMap>(file_reader, "cubemap/1"));

  std::uniform_real_distribution<float> u_dist(0.f, 1.f);

  for (const auto &[name, shape]: tank_factory->load_ammu_shapes()) {
    glm::vec4 color(u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, 1.f);

    player_renderer->add_drawable(
      name, std::make_unique<Specular>(
              file_reader, shape->get_vertices(), shape->get_normals(), color, color, color, 50.f,
              shape->get_id()));
  }

  for (const auto &item: engine->get_items()) {
    glm::vec4 color(u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, 1.f);

    player_renderer->add_drawable(
      item->get_name(),
      std::make_unique<Specular>(
        file_reader, item->get_shape()->get_vertices(), item->get_shape()->get_normals(), color,
        color, color, 50.f, item->get_shape()->get_id()));
  }

  for (auto &hud_drawable: player_controller_engine->get_hud_drawables(file_reader))
    player_renderer->add_hud_drawable(std::move(hud_drawable));
}

void UserGameTanksEnvironment::pause() { is_paused = true; }

UserGameTanksEnvironment::~UserGameTanksEnvironment() {
  player_renderer = std::nullptr_t();
  player_controller_engine = std::nullptr_t();
  tank_factory = std::nullptr_t();
}
