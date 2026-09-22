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
#include <arenai_view/camera.h>

#include "./item.h"

namespace arenai::model {

    struct ShellContactInfo {
        glm::vec3 fire_position;
        glm::vec3 current_position;
    };

    class Tank {
    public:
        virtual ~Tank() = default;

        virtual std::shared_ptr<view::AbstractCamera> get_camera() const = 0;
        virtual std::vector<std::shared_ptr<Item>> get_items() const = 0;
        virtual std::vector<std::shared_ptr<controller::Controller>> get_controllers() const = 0;

        virtual std::map<std::string, std::shared_ptr<Shape>> load_shell_shapes() const = 0;

        virtual bool is_dead() const = 0;

        virtual std::shared_ptr<Item> get_chassis() const = 0;
        virtual std::shared_ptr<Item> get_canon() const = 0;
    };

    struct RewardDetail {
        float aim = 0.f;
        float hit = 0.f;
        float received = 0.f;
        float death = 0.f;

        int nb_landed_shells = 0;
        float sum_aim_quality = 0.f;
        float sum_miss_distance = 0.f;
    };

    class EnemyTank : virtual public Tank {
    public:
        virtual float get_reward() const = 0;

        virtual RewardDetail get_last_reward_detail() const = 0;
        virtual std::vector<float> get_proprioception() const = 0;

        virtual void tick(const std::vector<std::shared_ptr<EnemyTank>> &tanks) = 0;

        virtual bool consume_has_hit() = 0;
        virtual bool consume_has_kill() = 0;
        virtual bool consume_has_fire() = 0;

        virtual bool is_first_frame_dead() const = 0;
        virtual bool is_suicide() const = 0;
        virtual bool is_timeout() const = 0;

        virtual void on_death() = 0;
    };

    struct PlayerHits {
        int hits = 0;
        int kills = 0;
    };

    class PlayerTank : virtual public Tank {
    public:
        virtual int get_score() const = 0;

        virtual PlayerHits consume_hits() = 0;
        virtual std::vector<ImpactInfo> consume_received_impacts() = 0;

        virtual void destroy() = 0;
    };

}// namespace arenai::model

#endif// ARENAI_TANK_H
