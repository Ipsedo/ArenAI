//
// Created by samuel on 29/09/2025.
//
#include <glm/gtc/matrix_transform.hpp>

#include <phyvr_core/environment.h>
#include <phyvr_model/convex.h>
#include <phyvr_model/height_map.h>
#include <phyvr_view/cubemap.h>
#include <phyvr_view/specular.h>

BaseTanksEnvironment::BaseTanksEnvironment(
  const std::shared_ptr<AbstractFileReader> &file_reader,
  const std::shared_ptr<AbstractGLContext> &gl_context, int nb_tanks, int threads_num)
    : nb_tanks(nb_tanks), threads_running(true),
      thread_barrier(std::make_unique<std::barrier<>>(static_cast<std::ptrdiff_t>(nb_tanks + 1))),
      tank_factories(), tank_renderers(), pool(),
      physic_engine(std::make_unique<PhysicEngine>(threads_num)), dev(), rng(dev()),
      file_reader(file_reader), gl_context(gl_context), enemy_visions() {}

std::vector<std::tuple<State, Reward, IsFinish>>
BaseTanksEnvironment::step(float time_delta, const std::vector<Action> &actions) {
  model_matrices.clear();

  for (auto &item: physic_engine->get_items())
    model_matrices.emplace_back(item->get_name(), item->get_model_matrix());

  model_matrices.emplace_back("cubemap", glm::scale(glm::mat4(1.), glm::vec3(2000., 2000., 2000.)));

  thread_barrier->arrive_and_wait();

  physic_engine->step(time_delta);
  on_draw(model_matrices);

  // TODO get enemy visions

  return std::vector<std::tuple<State, Reward, IsFinish>>(nb_tanks);
}

std::vector<State> BaseTanksEnvironment::reset_physics() {
  physic_engine->remove_bodies_and_constraints();
  tank_factories.clear();

  auto map = std::make_shared<HeightMapItem>(
    "height_map", file_reader, "heightmap/heightmap6.png", glm::vec3(0., 40., 0.),
    glm::vec3(10., 200., 10.));

  physic_engine->add_item(map);

  std::uniform_real_distribution<float> pos_u_dist(-500, 500);
  std::uniform_real_distribution<float> scale_u_dist(2.5, 10);
  std::uniform_real_distribution<float> mass_u_dist(3, 100);

  // add tanks
  for (int i = 0; i < nb_tanks; i++) {
    tank_factories.push_back(std::make_unique<TankFactory>(
      file_reader, "enemy_" + std::to_string(i),
      glm::vec3(pos_u_dist(rng), 20.f, pos_u_dist(rng))));

    for (const auto &item: tank_factories.back()->get_items()) physic_engine->add_item(item);
    for (const auto &item_producer: tank_factories.back()->get_item_producers())
      physic_engine->add_item_producer(item_producer);
  }

  // add basic shapes
  int nb_shapes = 5;

  for (int i = 0; i < nb_shapes; i++) {
    glm::vec3 pos(pos_u_dist(rng), 0.f, pos_u_dist(rng));
    glm::vec3 scale(scale_u_dist(rng));
    physic_engine->add_item(std::make_shared<SphereItem>(
      "sphere_" + std::to_string(i), file_reader, pos, scale, mass_u_dist(rng)));

    pos = glm::vec3(pos_u_dist(rng), 0.f, pos_u_dist(rng));
    scale = glm::vec3(scale_u_dist(rng));
    physic_engine->add_item(std::make_shared<CubeItem>(
      "cube_" + std::to_string(i), file_reader, pos, scale, mass_u_dist(rng)));

    pos = glm::vec3(pos_u_dist(rng), 0.f, pos_u_dist(rng));
    scale = glm::vec3(scale_u_dist(rng));
    physic_engine->add_item(std::make_shared<TetraItem>(
      "tetra_" + std::to_string(i), file_reader, pos, scale, mass_u_dist(rng)));

    pos = glm::vec3(pos_u_dist(rng), 0.f, pos_u_dist(rng));
    scale = glm::vec3(scale_u_dist(rng));
    physic_engine->add_item(std::make_shared<CylinderItem>(
      "cylinder_" + std::to_string(i), file_reader, pos, scale, mass_u_dist(rng)));
  }

  on_reset_physics(physic_engine);

  physic_engine->step(1.f / 60.f);// TODO attribute

  return std::vector<State>(nb_tanks);
}

void BaseTanksEnvironment::reset_drawables(
  const std::shared_ptr<AbstractGLContext> &new_gl_context) {
  kill_threads();

  tank_renderers.clear();
  gl_context = new_gl_context;

  on_reset_drawables(physic_engine, gl_context);

  std::uniform_real_distribution<float> u_dist(0.f, 1.f);

  std::uniform_real_distribution<float> light_pos_u_dist(-400.f, 400.f);
  std::uniform_real_distribution<float> light_height_u_dist(40.f, 400.f);

  for (auto &tank_factory: tank_factories)
    tank_renderers.push_back(std::make_unique<PBufferRenderer>(
      128, 128, glm::vec3(light_pos_u_dist(rng), light_height_u_dist(rng), light_pos_u_dist(rng)),
      tank_factory->get_camera()));

  glm::vec4 color(u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, 1.f);

  for (auto &tank_renderer: tank_renderers) {
    tank_renderer->add_drawable("cubemap", std::make_unique<CubeMap>(file_reader, "cubemap/1"));

    for (const auto &item: physic_engine->get_items())
      tank_renderer->add_drawable(
        item->get_name(),
        std::make_unique<Specular>(
          file_reader, item->get_shape()->get_vertices(), item->get_shape()->get_normals(), color,
          color, color, 50.f, item->get_shape()->get_id()));

    for (const auto &[name, shape]: tank_factories[0]->load_ammu_shapes()) {
      glm::vec4 shell_color(u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, 1.f);

      tank_renderer->add_drawable(
        name, std::make_unique<Specular>(
                file_reader, shape->get_vertices(), shape->get_normals(), shell_color, shell_color,
                shell_color, 50.f, shape->get_id()));
    }
  }

  start_threads();
}

void BaseTanksEnvironment::worker_enemy_vision(
  int index, std::unique_ptr<PBufferRenderer> &renderer) {

  enemy_visions[index] = renderer->draw_and_get_frame(model_matrices);

  bool is_running = true;

  while (is_running) {
    thread_barrier->arrive_and_wait();

    if (!threads_running.load(std::memory_order_acquire)) is_running = false;

    enemy_visions[index] = renderer->draw_and_get_frame(model_matrices);

    std::this_thread::sleep_for(std::chrono::duration<float>(1.f / 90.f));
  }
}

void BaseTanksEnvironment::start_threads() {
  const auto participants = static_cast<std::ptrdiff_t>(tank_renderers.size() + 1);
  thread_barrier = std::make_unique<std::barrier<>>(participants);

  enemy_visions.clear();
  enemy_visions.resize(tank_renderers.size());

  threads_running.store(true, std::memory_order_release);
  pool.clear();
  pool.reserve(tank_renderers.size());
  for (int i = 0; i < tank_renderers.size(); ++i)
    pool.emplace_back([this, i]() { worker_enemy_vision(i, tank_renderers[i]); });
}

void BaseTanksEnvironment::kill_threads() {
  if (pool.empty()) return;

  threads_running.store(false, std::memory_order_release);
  thread_barrier->arrive_and_wait();
  for (auto &t: pool)
    if (t.joinable()) t.join();

  pool.clear();
}

BaseTanksEnvironment::~BaseTanksEnvironment() {
  kill_threads();

  tank_factories.clear();
  tank_renderers.clear();
  enemy_visions.clear();
  model_matrices.clear();
}
