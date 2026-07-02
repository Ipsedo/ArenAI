//
// Created by samuel on 29/09/2025.
//
#include <cmath>
#include <iostream>
#include <thread>

#include <glm/gtc/matrix_transform.hpp>

#include <arenai_core/constants.h>
#include <arenai_core/environment.h>
#include <arenai_model/item_factory.h>
#include <arenai_model/tank_factory.h>

using namespace arenai;
using namespace arenai::core;

namespace arenai::core {

    BaseTanksEnvironment::BaseTanksEnvironment(
        const std::shared_ptr<utils::AbstractFileReader> &file_reader,
        const std::shared_ptr<view::AbstractGLContext> &gl_context, const int nb_tanks,
        float wanted_frequency, const int vision_height, const int vision_width,
        const int vision_num_threads, const bool vision_thread_sleep)
        : wanted_frequency(wanted_frequency), nb_tanks(nb_tanks), vision_height(vision_height),
          vision_width(vision_width), vision_num_threads(vision_num_threads),
          vision_thread_sleep(vision_thread_sleep),
          physic_engine(model::make_physic_engine(wanted_frequency)),
          nb_reset_frames(static_cast<int>(4.f / wanted_frequency)), drawing_started_(false),
          gl_context(gl_context), rng(dev()), file_reader(file_reader),
          vision_pool_(std::make_unique<EnemyVisionThreadPool>(
              nb_tanks, vision_num_threads, vision_height, vision_width, wanted_frequency,
              vision_thread_sleep)) {}

    std::vector<std::tuple<State, Reward, IsDone>>
    BaseTanksEnvironment::step(const float time_delta, const std::vector<Action> &actions) {

        // 1. apply action
        for (int i = 0; i < tank_factories.size(); i++) {
            if (!tank_factories[i]->is_dead()) tank_controller_handler[i]->on_event(actions[i]);
            else tank_factories[i]->on_death();
        }

        // 2. step physic
        physic_engine->step(time_delta);

        // 3. set model matrices double buffer and draw scene
        const auto curr_model_matrices = get_model_matrices();
        vision_pool_->set_model_matrices(curr_model_matrices);

        on_draw(curr_model_matrices);
        vision_pool_->loop_wait();

        // 4. build State
        std::vector<std::tuple<State, Reward, IsDone>> result;
        result.reserve(tank_factories.size());

        for (int i = 0; i < tank_factories.size(); i++) {
            result.emplace_back(
                State(vision_pool_->read_vision(i), tank_factories[i]->get_proprioception()),
                tank_factories[i]->get_reward(tank_factories), tank_factories[i]->is_dead());
        }

        return result;
    }

    void BaseTanksEnvironment::reset_physics(const float spawn_width, const float spawn_height) {
        physic_engine->remove_bodies_and_constraints();
        tank_controller_handler.clear();
        tank_factories.clear();

        auto item_factory = physic_engine->get_item_factory();
        auto tank_factory = model::make_tank_factory(*physic_engine, file_reader, wanted_frequency);

        item_factory->make_height_map_item(
            "height_map", file_reader, "heightmap/heightmap6.png", glm::vec3(0., 40., 0.),
            glm::vec3(10., 200., 10.));

        std::uniform_real_distribution<float> x_pos_u_dist(-spawn_width / 2, spawn_width / 2);
        std::uniform_real_distribution<float> y_pos_u_dist(-spawn_height / 2, spawn_height / 2);

        std::uniform_real_distribution<float> mass_u_dist(3, 100);

        // add tanks
        for (int i = 0; i < nb_tanks; i++) {
            tank_factories.push_back(tank_factory->make_enemy_tank(
                "enemy_" + std::to_string(i),
                glm::vec3(x_pos_u_dist(rng), 0.f, y_pos_u_dist(rng))));

            tank_controller_handler.push_back(std::make_unique<EnemyControllerHandler>(
                wanted_frequency, 1.f / 6.f, tank_factories.back()->get_action_stats(),
                model::ENEMY_TURRET_RADIAL_VELOCITY));

            for (const auto &controller: tank_factories.back()->get_controllers())
                tank_controller_handler.back()->add_controller(controller);
        }

        // add basic shapes
        std::uniform_real_distribution<float> scale_u_dist(2.5, 10);
        constexpr int nb_shapes = 5;

        for (int i = 0; i < nb_shapes; i++) {
            glm::vec3 pos(x_pos_u_dist(rng), 0.f, y_pos_u_dist(rng));
            glm::vec3 scale(scale_u_dist(rng));
            item_factory->make_sphere_item(
                "sphere_" + std::to_string(i), file_reader, pos, scale, mass_u_dist(rng));

            pos = glm::vec3(x_pos_u_dist(rng), 0.f, y_pos_u_dist(rng));
            scale = glm::vec3(scale_u_dist(rng));
            item_factory->make_cube_item(
                "cube_" + std::to_string(i), file_reader, pos, scale, mass_u_dist(rng));

            pos = glm::vec3(x_pos_u_dist(rng), 0.f, y_pos_u_dist(rng));
            scale = glm::vec3(scale_u_dist(rng));
            item_factory->make_tetra_item(
                "tetra_" + std::to_string(i), file_reader, pos, scale, mass_u_dist(rng));

            pos = glm::vec3(x_pos_u_dist(rng), 0.f, y_pos_u_dist(rng));
            scale = glm::vec3(scale_u_dist(rng));
            item_factory->make_cylinder_item(
                "cylinder_" + std::to_string(i), file_reader, pos, scale, mass_u_dist(rng));
        }

        on_reset_physics(physic_engine);

        for (int i = 0; i < nb_reset_frames; i++) physic_engine->step(wanted_frequency);
    }

    std::vector<State>
    BaseTanksEnvironment::reset(const float spawn_width, const float spawn_height) {
        if (drawing_started_) stop_drawing();

        reset_physics(spawn_width, spawn_height);
        reset_drawables();
        drawing_started_ = true;

        // warm-up: 2 render loops to prime PBO double-buffering
        const auto model_matrices = get_model_matrices();
        for (int i = 0; i < 2; i++) {
            vision_pool_->set_model_matrices(model_matrices);
            on_draw(model_matrices);
            vision_pool_->loop_wait();
        }

        std::vector<State> states;
        states.reserve(tank_factories.size());
        for (int i = 0; i < static_cast<int>(tank_factories.size()); i++)
            states.emplace_back(
                vision_pool_->read_vision(i), tank_factories[i]->get_proprioception());

        return states;
    }

    void BaseTanksEnvironment::reset_drawables(
        const std::shared_ptr<view::AbstractGLContext> &new_gl_context) {
        gl_context = new_gl_context;
        gl_context->make_current();

        on_reset_drawables(physic_engine, gl_context);

        gl_context->release_current();

        vision_pool_->start_thread(
            tank_factories, gl_context, file_reader, get_model_matrices(),
            physic_engine->get_items());

        gl_context->make_current();
    }

    void BaseTanksEnvironment::reset_drawables() { reset_drawables(gl_context); }

    void BaseTanksEnvironment::seed(const unsigned int seed) {
        rng.seed(seed);
        vision_pool_->set_seed(seed);
    }

    void BaseTanksEnvironment::stop_drawing() const { vision_pool_->kill_threads(); }

    std::vector<std::tuple<std::string, glm::mat4>>
    BaseTanksEnvironment::get_model_matrices() const {
        std::vector<std::tuple<std::string, glm::mat4>> curr_model_matrices;

        const auto items = physic_engine->get_items();
        curr_model_matrices.reserve(items.size() + 1);
        for (const auto &item: items)
            curr_model_matrices.emplace_back(item->get_name(), item->get_model_matrix());

        curr_model_matrices.emplace_back("cubemap", glm::scale(glm::mat4(1.f), glm::vec3(2000.f)));

        return curr_model_matrices;
    }

    BaseTanksEnvironment::~BaseTanksEnvironment() {
        stop_drawing();
        tank_factories.clear();
    }

}// namespace arenai::core
