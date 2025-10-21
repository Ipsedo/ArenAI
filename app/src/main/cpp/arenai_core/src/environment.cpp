//
// Created by samuel on 29/09/2025.
//
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <thread>

#include <glm/gtc/matrix_transform.hpp>

#include <arenai_core/constants.h>
#include <arenai_core/environment.h>
#include <arenai_model/convex.h>
#include <arenai_model/height_map.h>
#include <arenai_view/cubemap.h>
#include <arenai_view/specular.h>

BaseTanksEnvironment::BaseTanksEnvironment(
    const std::shared_ptr<AbstractFileReader> &file_reader,
    const std::shared_ptr<AbstractGLContext> &gl_context, const int nb_tanks,
    float wanted_frequency, const bool thread_sleep)
    : wanted_frequency(wanted_frequency), nb_tanks(nb_tanks), visions_mutex(nb_tanks),
      thread_killed(true), thread_sleep(thread_sleep), threads_running(true),
      thread_barrier(std::make_unique<std::barrier<>>(static_cast<std::ptrdiff_t>(nb_tanks + 1))),
      pool(), tank_factories(), tank_controller_handler(),
      enemy_visions(
          nb_tanks, image<uint8_t>(
                        3, std::vector<std::vector<uint8_t>>(
                               ENEMY_VISION_SIZE, std::vector<uint8_t>(ENEMY_VISION_SIZE, 0)))),
      physic_engine(std::make_unique<PhysicEngine>(wanted_frequency)), gl_context(gl_context),
      dev(), rng(dev()), file_reader(file_reader) {

    std::uniform_int_distribution<u_int8_t> u_dist(0, 255);
    for (int i = 0; i < nb_tanks; i++)
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < ENEMY_VISION_SIZE; h++)
                for (int w = 0; w < ENEMY_VISION_SIZE; w++) enemy_visions[i][c][h][w] = u_dist(rng);
}

std::vector<std::tuple<State, Reward, IsFinish>> BaseTanksEnvironment::step(
    const float time_delta, std::future<std::vector<Action>> &actions_future) {

    {
        lock_all_thread_model_matrix();
        model_matrices.clear();

        for (const auto &item: physic_engine->get_items())
            model_matrices.emplace_back(item->get_name(), item->get_model_matrix());

        model_matrices.emplace_back(
            "cubemap", glm::scale(glm::mat4(1.), glm::vec3(2000., 2000., 2000.)));
    }

    thread_barrier->arrive_and_wait();

    physic_engine->step(time_delta);
    on_draw(model_matrices);

    std::vector<std::tuple<State, Reward, IsFinish>> result;
    result.reserve(tank_factories.size());

    for (int i = 0; i < tank_factories.size(); i++) {
        std::lock_guard lock_guard(visions_mutex[i]);
        result.emplace_back(
            State(enemy_visions[i], tank_factories[i]->get_proprioception()),
            tank_factories[i]->get_reward(), tank_factories[i]->is_dead());
    }

    const auto actions = actions_future.get();
    for (int i = 0; i < tank_factories.size(); i++) {
        if (!tank_factories[i]->is_dead()) tank_controller_handler[i]->on_event(actions[i]);
        else
            for (const auto &item: tank_factories[i]->dead_and_get_items())
                physic_engine->remove_item_constraints_from_world(item);
    }

    return result;
}

std::vector<State> BaseTanksEnvironment::reset_physics() {
    physic_engine->remove_bodies_and_constraints();
    tank_controller_handler.clear();
    tank_factories.clear();

    const auto map = std::make_shared<HeightMapItem>(
        "height_map", file_reader, "heightmap/heightmap6.png", glm::vec3(0., 40., 0.),
        glm::vec3(10., 200., 10.));

    physic_engine->add_item(map);

    std::uniform_real_distribution<float> pos_u_dist(-500, 500);
    std::uniform_real_distribution<float> scale_u_dist(2.5, 10);
    std::uniform_real_distribution<float> mass_u_dist(3, 100);

    // add tanks
    for (int i = 0; i < nb_tanks; i++) {
        tank_factories.push_back(std::make_unique<EnemyTankFactory>(
            file_reader, "enemy_" + std::to_string(i),
            glm::vec3(pos_u_dist(rng), 0.f, pos_u_dist(rng)), wanted_frequency));

        for (const auto &item: tank_factories.back()->get_items()) physic_engine->add_item(item);
        for (const auto &item_producer: tank_factories.back()->get_item_producers())
            physic_engine->add_item_producer(item_producer);

        tank_controller_handler.push_back(std::make_unique<EnemyControllerHandler>(
            wanted_frequency, 1.f / 6.f, tank_factories.back()->get_action_stats()));

        for (const auto &controller: tank_factories.back()->get_controllers())
            tank_controller_handler.back()->add_controller(controller);
    }

    // add basic shapes
    constexpr int nb_shapes = 5;

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

    physic_engine->step(wanted_frequency);

    std::vector<State> states;
    for (int i = 0; i < tank_factories.size(); i++) {
        std::lock_guard lock_guard(visions_mutex[i]);
        states.emplace_back(enemy_visions[i], tank_factories[i]->get_proprioception());
    }

    {
        lock_all_thread_model_matrix();
        model_matrices.clear();

        for (const auto &item: physic_engine->get_items())
            model_matrices.emplace_back(item->get_name(), item->get_model_matrix());

        model_matrices.emplace_back(
            "cubemap", glm::scale(glm::mat4(1.), glm::vec3(2000., 2000., 2000.)));
    }

    return states;
}

void BaseTanksEnvironment::reset_drawables(
    const std::shared_ptr<AbstractGLContext> &new_gl_context) {
    if (!thread_killed) kill_threads();

    gl_context = new_gl_context;
    gl_context->make_current();

    on_reset_drawables(physic_engine, gl_context);

    gl_context->release_current();

    start_threads();

    gl_context->make_current();
}

void BaseTanksEnvironment::stop_drawing() {
    if (!thread_killed) kill_threads();
}

void BaseTanksEnvironment::worker_enemy_vision(
    const int index, const std::unique_ptr<EnemyTankFactory> &tank_factory) {
    std::mt19937 thread_rng;

    auto renderer = std::make_unique<PBufferRenderer>(
        gl_context, ENEMY_VISION_SIZE, ENEMY_VISION_SIZE, glm::vec3(200, 300, 200),
        tank_factory->get_camera());

    renderer->make_current();

    std::uniform_real_distribution<float> u_dist(0.f, 1.f);

    renderer->add_drawable("cubemap", std::make_unique<CubeMap>(file_reader, "cubemap/1"));

    for (const auto &item: physic_engine->get_items()) {

        glm::vec4 color(u_dist(thread_rng), u_dist(thread_rng), u_dist(thread_rng), 1.f);
        renderer->add_drawable(
            item->get_name(),
            std::make_unique<Specular>(
                file_reader, item->get_shape()->get_vertices(), item->get_shape()->get_normals(),
                color, color, color, 50.f, item->get_shape()->get_id()));
    }

    for (const auto &[name, shape]: tank_factories[0]->load_shell_shapes()) {
        glm::vec4 shell_color(u_dist(thread_rng), u_dist(thread_rng), u_dist(thread_rng), 1.f);

        renderer->add_drawable(
            name, std::make_unique<Specular>(
                      file_reader, shape->get_vertices(), shape->get_normals(), shell_color,
                      shell_color, shell_color, 50.f, shape->get_id()));
    }

    const auto frame_dt = std::chrono::milliseconds(static_cast<int>(wanted_frequency * 1000.f));
    auto last_time = std::chrono::steady_clock::now();

    for (;;) {
        thread_barrier->arrive_and_wait();

        if (!threads_running.load(std::memory_order_acquire)) break;

        auto now = std::chrono::steady_clock::now();
        auto dt = now - last_time;
        last_time = now;

        {
            std::lock_guard lock(visions_mutex[index]);
            enemy_visions[index] = renderer->draw_and_get_frame(model_matrices);
        }

        if (thread_sleep) std::this_thread::sleep_for(frame_dt - dt);
    }

    renderer.reset();
    eglReleaseThread();
}

void BaseTanksEnvironment::lock_all_thread_model_matrix() {
    std::vector<std::unique_lock<std::mutex>> locks;
    locks.reserve(visions_mutex.size());

    std::vector<std::mutex *> ptrs;
    for (auto &m: visions_mutex) ptrs.push_back(&m);

    std::sort(ptrs.begin(), ptrs.end(), std::less<std::mutex *>{});
    ptrs.erase(std::unique(ptrs.begin(), ptrs.end()), ptrs.end());

    locks.reserve(ptrs.size());
    for (auto *pm: ptrs) locks.emplace_back(*pm);
}

void BaseTanksEnvironment::start_threads() {
    const auto participants = static_cast<std::ptrdiff_t>(tank_factories.size() + 1);
    thread_barrier = std::make_unique<std::barrier<>>(participants);

    enemy_visions = std::vector(
        nb_tanks,
        image<uint8_t>(
            3, std::vector(ENEMY_VISION_SIZE, std::vector<uint8_t>(ENEMY_VISION_SIZE, 0))));

    threads_running.store(true, std::memory_order_release);
    pool.clear();
    pool.reserve(tank_factories.size());

    for (int i = 0; i < tank_factories.size(); ++i)
        pool.emplace_back([this, i]() { worker_enemy_vision(i, tank_factories[i]); });

    thread_killed = false;
}

void BaseTanksEnvironment::kill_threads() {
    if (pool.empty()) return;

    threads_running.store(false, std::memory_order_release);
    thread_barrier->arrive_and_wait();
    for (auto &t: pool)
        if (t.joinable()) t.join();

    pool.clear();

    thread_killed = true;
}

BaseTanksEnvironment::~BaseTanksEnvironment() {
    kill_threads();

    tank_factories.clear();
    enemy_visions.clear();
    model_matrices.clear();
}
