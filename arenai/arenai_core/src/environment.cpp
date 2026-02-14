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

void VisionDoubleBuffer::init(std::mt19937 &rng, const int height, const int width) {
    std::uniform_int_distribution<u_int8_t> u_dist(0, 255);

    for (int i = 0; i < 2; i++) {
        const auto size = 3 * height * width;
        buf[i] = std::make_shared<image<uint8_t>>(std::vector<uint8_t>(size));

        for (int j = 0; j < size; j++) buf[i]->pixels[j] = u_dist(rng);
    }
}

BaseTanksEnvironment::BaseTanksEnvironment(
    const std::shared_ptr<AbstractFileReader> &file_reader,
    const std::shared_ptr<AbstractGLContext> &gl_context, const int nb_tanks,
    float wanted_frequency, const bool thread_sleep)
    : wanted_frequency(wanted_frequency), nb_tanks(nb_tanks), thread_sleep(thread_sleep),
      threads_running(false), enemy_visions(nb_tanks),
      physic_engine(std::make_unique<PhysicEngine>(wanted_frequency)), gl_context(gl_context),
      nb_reset_frames(static_cast<int>(4.f / wanted_frequency)), rng(dev()),
      file_reader(file_reader) {

    for (auto &buffer: enemy_visions) buffer.init(rng, ENEMY_VISION_HEIGHT, ENEMY_VISION_WIDTH);
}

std::vector<std::tuple<State, Reward, IsDone>> BaseTanksEnvironment::step(
    const float time_delta, std::future<std::vector<Action>> &actions_future) {

    // set model matrices double buffer
    const auto curr_model_matrices = publish_and_get_model_matrices();

    // step physics
    physic_engine->step(time_delta);
    on_draw(curr_model_matrices);

    // build State
    std::vector<std::tuple<State, Reward, IsDone>> result;
    result.reserve(tank_factories.size());

    for (int i = 0; i < tank_factories.size(); i++) {
        auto &[buf, front] = enemy_visions[i];
        const int f = front.load(std::memory_order_acquire);
        const auto &img = buf[f];

        result.emplace_back(
            State(img, tank_factories[i]->get_proprioception()),
            tank_factories[i]->get_reward(tank_factories), tank_factories[i]->is_dead());
    }

    // apply actions
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
    if (threads_running.load(std::memory_order_acquire)) kill_threads();

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

    for (int i = 0; i < nb_reset_frames; i++) physic_engine->step(wanted_frequency);

    std::vector<State> states;
    for (int i = 0; i < tank_factories.size(); i++) {
        auto &[buf, front] = enemy_visions[i];
        int f = front.load(std::memory_order_acquire);
        const auto &img = buf[f];

        states.emplace_back(img, tank_factories[i]->get_proprioception());
    }

    publish_and_get_model_matrices();

    return states;
}

void BaseTanksEnvironment::reset_drawables(
    const std::shared_ptr<AbstractGLContext> &new_gl_context) {
    if (threads_running.load(std::memory_order_acquire)) kill_threads();

    gl_context = new_gl_context;
    gl_context->make_current();

    on_reset_drawables(physic_engine, gl_context);

    gl_context->release_current();

    start_threads();

    gl_context->make_current();
}

void BaseTanksEnvironment::stop_drawing() {
    if (threads_running.load(std::memory_order_acquire)) kill_threads();
}

void BaseTanksEnvironment::worker_enemy_vision(
    const int index, const std::unique_ptr<EnemyTankFactory> &tank_factory) {
    uint64_t seen = model_matrices.frame_id.load(std::memory_order_acquire);

    auto renderer = std::make_unique<PBufferRenderer>(
        gl_context, ENEMY_VISION_WIDTH, ENEMY_VISION_HEIGHT, glm::vec3(200, 300, 200),
        tank_factory->get_camera());

    renderer->make_current();

    std::uniform_real_distribution u_dist(0.f, 1.f);

    renderer->add_drawable("cubemap", std::make_unique<CubeMap>(file_reader, "cubemap/1"));

    for (const auto &item: physic_engine->get_items()) {
        glm::vec4 color(u_dist(rng), u_dist(rng), u_dist(rng), 1.f);
        renderer->add_drawable(
            item->get_name(),
            std::make_unique<Specular>(
                file_reader, item->get_shape()->get_vertices(), item->get_shape()->get_normals(),
                color, color, color, 50.f, item->get_shape()->get_id()));
    }

    for (const auto &[name, shape]: tank_factories[0]->load_shell_shapes()) {
        glm::vec4 shell_color(u_dist(rng), u_dist(rng), u_dist(rng), 1.f);

        renderer->add_drawable(
            name, std::make_unique<Specular>(
                      file_reader, shape->get_vertices(), shape->get_normals(), shell_color,
                      shell_color, shell_color, 50.f, shape->get_id()));
    }

    const auto frame_dt = std::chrono::milliseconds(static_cast<int>(wanted_frequency * 1000.f));
    auto last_time = std::chrono::steady_clock::now();

    while (threads_running.load(std::memory_order_acquire)) {
        auto now = std::chrono::steady_clock::now();
        auto dt = now - last_time;
        last_time = now;

        model_matrices.frame_id.wait(seen, std::memory_order_acquire);

        const uint64_t new_seen = model_matrices.frame_id.load(std::memory_order_acquire);
        seen = new_seen;

        const int f = model_matrices.front_idx.load(std::memory_order_acquire);
        const auto &matrices = model_matrices.buf[f];

        auto &[buf, front] = enemy_visions[index];
        const int back = 1 - front.load(std::memory_order_relaxed);

        buf[back] = renderer->draw_and_get_frame(matrices);
        front.store(back, std::memory_order_release);

        if (thread_sleep) std::this_thread::sleep_for(frame_dt - dt);
    }

    renderer.reset();
    eglReleaseThread();
}

void BaseTanksEnvironment::start_threads() {
    enemy_visions = std::vector<VisionDoubleBuffer>(nb_tanks);
    for (auto &buffer: enemy_visions) buffer.init(rng, ENEMY_VISION_HEIGHT, ENEMY_VISION_WIDTH);

    threads_running.store(true, std::memory_order_release);
    pool.clear();
    pool.reserve(tank_factories.size());

    for (int i = 0; i < tank_factories.size(); ++i)
        pool.emplace_back([this, i] { worker_enemy_vision(i, tank_factories[i]); });
}

void BaseTanksEnvironment::kill_threads() {
    if (pool.empty()) return;

    threads_running.store(false, std::memory_order_release);

    model_matrices.frame_id.fetch_add(1, std::memory_order_release);
    model_matrices.frame_id.notify_all();

    for (auto &t: pool)
        if (t.joinable()) t.join();

    pool.clear();
}

std::vector<std::tuple<std::string, glm::mat4>>
BaseTanksEnvironment::publish_and_get_model_matrices() {
    const int back = 1 - model_matrices.front_idx.load(std::memory_order_relaxed);
    auto &dst = model_matrices.buf[back];

    const auto items = physic_engine->get_items();

    dst.clear();
    dst.reserve(items.size() + 1);

    for (const auto &item: items) dst.emplace_back(item->get_name(), item->get_model_matrix());

    dst.emplace_back("cubemap", glm::scale(glm::mat4(1.f), glm::vec3(2000.f)));

    model_matrices.front_idx.store(back, std::memory_order_release);
    model_matrices.frame_id.fetch_add(1, std::memory_order_release);
    model_matrices.frame_id.notify_all();

    return dst;
}

BaseTanksEnvironment::~BaseTanksEnvironment() {
    stop_drawing();

    tank_factories.clear();
    enemy_visions.clear();
}
