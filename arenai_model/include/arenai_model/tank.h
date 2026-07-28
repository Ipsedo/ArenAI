//
// Created by samuel on 28/09/2025.
//

#ifndef ARENAI_TANK_H
#define ARENAI_TANK_H

#include <map>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include <arenai_controller/controller.h>
#include <arenai_utils/file_reader.h>
#include <arenai_view/camera.h>

#include "./item.h"

namespace arenai::model {

    struct ShellContactInfo {
        glm::vec3 fire_position;
        glm::vec3 current_position;
    };

    class ActionStats;

    class Tank {
    public:
        virtual ~Tank() = default;

        virtual std::shared_ptr<view::AbstractCamera> get_camera() = 0;
        virtual std::vector<std::shared_ptr<Item>> get_items() = 0;
        virtual std::vector<std::shared_ptr<controller::Controller>> get_controllers() = 0;
        virtual std::map<std::string, std::shared_ptr<Shape>> load_shell_shapes() const = 0;
        virtual bool is_dead() = 0;
        virtual std::shared_ptr<Item> get_chassis() = 0;
        virtual std::shared_ptr<Item> get_canon() = 0;
    };

    class EnemyTank : virtual public Tank {
    public:
        virtual float get_reward(const std::vector<std::shared_ptr<EnemyTank>> &tanks) = 0;
        virtual float get_phi(const std::vector<std::shared_ptr<EnemyTank>> &tanks) = 0;
        virtual std::vector<float> get_proprioception() = 0;
        virtual std::shared_ptr<ActionStats> get_action_stats() = 0;
        virtual bool has_hit_other_tank() = 0;
        virtual bool is_suicide() const = 0;
        virtual bool is_first_frame_dead() = 0;
        virtual void on_death() = 0;
    };

    class PlayerTank : virtual public Tank {
    public:
        virtual int get_score() const = 0;
    };

}// namespace arenai::model

#endif// ARENAI_TANK_H
