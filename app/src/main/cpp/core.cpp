//
// Created by samuel on 21/03/2023.
//

#include "./core.h"
#include "./model/items/convex.h"
#include "./model/items/height_map.h"
#include "./model/shapes.h"
#include "./model/tank/tank_factory.h"
#include "./utils/logging.h"
#include "./view/drawable/cubemap.h"
#include "./view/drawable/specular.h"

#include <glm/gtx/transform.hpp>

CoreEngine::CoreEngine(struct android_app *app)
    : tank_factory(app->activity->assetManager, glm::vec3(0., -40., 40)),
      is_paused(true), camera(std::nullptr_t()), renderer(std::nullptr_t()),
      physic_engine(std::make_unique<PhysicEngine>()),
      controller_engine(std::nullptr_t()), dev(), rng(dev()) {

  auto map = std::make_shared<HeightMapItem>(
      "height_map", app->activity->assetManager, "heightmap/heightmap6.png",
      glm::vec3(0., 40., 0.), glm::vec3(10., 200., 10.));

  physic_engine->add_item(map);

  for (auto &item : tank_factory.get_items()) {
    physic_engine->add_item(item);
  }

  for (auto &item_producer : tank_factory.get_item_producers())
    physic_engine->add_item_producer(item_producer);

  camera = tank_factory.get_camera();

  // add basic shapes
  int nb_shapes = 5;

  std::uniform_real_distribution<float> pos_u_dist(-500, 500);
  std::uniform_real_distribution<float> scale_u_dist(2.5, 10);
  std::uniform_real_distribution<float> mass_u_dist(3, 100);

  for (int i = 0; i < nb_shapes; i++) {
    glm::vec3 pos(pos_u_dist(rng), 0.f, pos_u_dist(rng));
    glm::vec3 scale(scale_u_dist(rng));
    physic_engine->add_item(std::make_shared<SphereItem>(
        "sphere_" + std::to_string(i), app->activity->assetManager, pos, scale,
        mass_u_dist(rng)));

    pos = glm::vec3(pos_u_dist(rng), 0.f, pos_u_dist(rng));
    scale = glm::vec3(scale_u_dist(rng));
    physic_engine->add_item(std::make_shared<CubeItem>(
        "cube_" + std::to_string(i), app->activity->assetManager, pos, scale,
        mass_u_dist(rng)));

    pos = glm::vec3(pos_u_dist(rng), 0.f, pos_u_dist(rng));
    scale = glm::vec3(scale_u_dist(rng));
    physic_engine->add_item(std::make_shared<TetraItem>(
        "tetra_" + std::to_string(i), app->activity->assetManager, pos, scale,
        mass_u_dist(rng)));

    pos = glm::vec3(pos_u_dist(rng), 0.f, pos_u_dist(rng));
    scale = glm::vec3(scale_u_dist(rng));
    physic_engine->add_item(std::make_shared<CylinderItem>(
        "cylinder_" + std::to_string(i), app->activity->assetManager, pos,
        scale, mass_u_dist(rng)));
  }
}

void CoreEngine::_new_view(AAssetManager *mgr, ANativeWindow *window,
                           AConfiguration *config) {
  renderer = std::make_unique<Renderer>(window, camera);
  controller_engine = std::make_unique<ControllerEngine>(
      config, renderer->get_width(), renderer->get_height());

  // init controller
  for (auto &ctrl : tank_factory.get_controllers())
    controller_engine->add_controller(ctrl);

  // init renderer
  std::uniform_real_distribution<float> u_dist(0., 1.);

  renderer->add_drawable("cubemap",
                         std::make_unique<CubeMap>(mgr, "cubemap/1"));

  for (const auto &item : physic_engine->get_items()) {
    glm::vec4 color(u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, u_dist(rng) * 0.8f,
                    1.f);

    renderer->add_drawable(
        item->get_name(),
        std::make_unique<Specular>(mgr, item->get_shape()->get_vertices(),
                                   item->get_shape()->get_normals(), color,
                                   color, color, 50.f));
  }

  for (const auto &[name, shape] : tank_factory.load_ammu_shapes()) {
    glm::vec4 color(u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, u_dist(rng) * 0.8f,
                    1.f);

    renderer->add_drawable(
        name, std::make_unique<Specular>(mgr, shape->get_vertices(),
                                         shape->get_normals(), color, color,
                                         color, 50.f));
  }

  for (auto &hud_drawable : controller_engine->get_hud_drawables(mgr))
    renderer->add_hud_drawable(std::move(hud_drawable));

  is_paused = false;
}

void CoreEngine::draw() {
  if (is_paused)
    return;

  auto model_matrices = std::vector<std::tuple<std::string, glm::mat4>>();

  for (auto &item : physic_engine->get_items())
    model_matrices.emplace_back(item->get_name(), item->get_model_matrix());

  model_matrices.emplace_back(
      "cubemap", glm::scale(glm::mat4(1.), glm::vec3(2000., 2000., 2000.)));

  renderer->draw(model_matrices);
}

void CoreEngine::step(float time_delta) {
  if (is_paused)
    return;

  physic_engine->step(time_delta);
}

int32_t CoreEngine::on_input(struct android_app *app, AInputEvent *event) {
  if (AKeyEvent_getKeyCode(event) == AKEYCODE_BACK) {
    ANativeActivity_finish(app->activity);
    return 1;
  }

  return controller_engine->on_event(event);
}

void CoreEngine::_pause() {
  is_paused = true;
  renderer = std::nullptr_t();
}

bool CoreEngine::is_running() const { return !is_paused; }

void CoreEngine::on_cmd(struct android_app *app, int32_t cmd) {
  switch (cmd) {
  case APP_CMD_SAVE_STATE:
    // The system has asked us to save our current state.  Do so.
    /*app->savedState = malloc(sizeof(CoreEngine));
    // TODO real object copy...
    app->savedState = engine;
    app->savedStateSize = sizeof(CoreEngine);
    */
    LOG_INFO("save state");
    break;
  case APP_CMD_INIT_WINDOW:
    // The window is being shown, get it ready.
    if (app->window != nullptr) {
      LOG_INFO("opening window");
      _new_view(app->activity->assetManager, app->window, app->config);
    }
    break;
  case APP_CMD_TERM_WINDOW:
    _pause();
    LOG_INFO("close");
    break;
  case APP_CMD_GAINED_FOCUS:
    LOG_INFO("gained focus");
    break;
  case APP_CMD_LOST_FOCUS:
    _pause();
    LOG_INFO("lost focus");
    /*engine_draw_frame(engine);*/
    break;
  default:
    break;
  }
}
