//
// Created by samuel on 02/04/2023.
//

#include "./jolt_tank.h"

#include <algorithm>
#include <atomic>

#include <Jolt/Physics/Collision/GroupFilterTable.h>

#include "../jolt_engine.h"
#include "./parts/canon.h"
#include "./parts/chassis.h"
#include "./parts/shell.h"
#include "./parts/turret.h"
#include "./parts/wheel.h"

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    JoltTank::JoltTank(
        JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::string &tank_prefix_name, glm::vec3 chassis_pos,
        const float wanted_frame_frequency,
        const std::function<void(const ShellItem *, const ShellContactInfo &, Item *)>
            &on_contact_callback,
        const std::function<void(const std::shared_ptr<ShellItem> &)> &on_shell_fired_callback,
        const std::function<bool()> &can_fire_callback)
        : engine(engine), name(tank_prefix_name), camera(std::nullptr_t()),
          file_reader(file_reader) {

        glm::vec3 scale(0.5);

        // chassis
        constexpr float chassis_mass = 1e4f;
        auto chassis_item = std::make_shared<ChassisItem>(
            tank_prefix_name, engine, file_reader, chassis_pos, scale, chassis_mass);

        chassis = chassis_item;
        items.push_back(chassis);
        jolt_items.push_back(chassis_item);
        life_items.push_back(chassis_item.get());

        // wheels
        constexpr float front_axle_z = 3.f;

        constexpr float wheel_mass = 150.f;
        glm::vec3 wheel_scale = scale * glm::vec3(1.3, 1.1, 1.1);

        std::vector<std::tuple<std::string, glm::vec3, float>> front_wheel_config{
            {"wheel_right_1", {-2.7, -1., front_axle_z}, 1.f},
            {"wheel_left_1", {2.7, -1., front_axle_z}, 1.f},
            {"wheel_right_2", {-2.7, -1., 0.}, 0.5f},
            {"wheel_left_2", {2.7, -1., 0.}, 0.5f}};

        for (auto &[wheel_name, wheel_pos, angle_factor]: front_wheel_config) {
            auto wheel = std::make_shared<DirectionalWheelItem>(
                std::format("{}_{}", tank_prefix_name, wheel_name), engine, file_reader,
                wheel_pos + chassis_pos, wheel_pos, wheel_scale, wheel_mass,
                chassis_item->get_body(), front_axle_z, angle_factor);

            jolt_items.push_back(wheel);
            items.push_back(wheel);
            controllers.push_back(wheel);
            life_items.push_back(wheel.get());
        }

        std::vector<std::tuple<std::string, glm::vec3>> wheel_config{
            {"wheel_right_3", {-2.7, -1., -front_axle_z}},
            {"wheel_left_3", {2.7, -1., -front_axle_z}}};

        for (auto &[wheel_name, wheel_pos]: wheel_config) {
            auto wheel = std::make_shared<WheelItem>(
                std::format("{}_{}", tank_prefix_name, wheel_name), engine, file_reader,
                wheel_pos + chassis_pos, wheel_pos, wheel_scale, wheel_mass,
                chassis_item->get_body(), front_axle_z);

            jolt_items.push_back(wheel);
            items.push_back(wheel);
            controllers.push_back(wheel);
            life_items.push_back(wheel.get());
        }

        // turret
        glm::vec3 turret_pos(0.f, 1.3f, 1.2f);
        glm::vec3 turret_scale(1.2f);
        auto turret = std::make_shared<TurretItem>(
            tank_prefix_name, engine, file_reader, chassis_pos + turret_pos, turret_pos,
            scale * turret_scale, 300, chassis_item->get_body());
        jolt_items.push_back(turret);
        items.push_back(turret);
        controllers.push_back(turret);
        life_items.push_back(turret.get());

        // canon
        glm::vec3 canon_pos(0.f, 0.5f, 1.7f);
        glm::vec3 canon_scale = turret_scale;
        auto canon_item = std::make_shared<CanonItem>(
            tank_prefix_name, engine, file_reader, chassis_pos + turret_pos + canon_pos, canon_pos,
            scale * canon_scale, 100, turret->get_body(), wanted_frame_frequency,
            [on_contact_callback](
                const ShellItem *shell, const glm::vec3 fire_pos, const glm::vec3 hit_pos,
                Item *item) {
                on_contact_callback(
                    shell, {.fire_position = fire_pos, .current_position = hit_pos}, item);
            },
            on_shell_fired_callback, can_fire_callback);

        jolt_items.push_back(canon_item);
        items.push_back(canon_item);
        controllers.push_back(canon_item);
        life_items.push_back(canon_item.get());

        canon = canon_item;

        // spring-arm camera: pull the canon camera in front of any world geometry
        // (terrain, obstacles) blocking the [aim point -> camera] segment. The
        // tank's own bodies are excluded, the ray starts inside them.
        std::vector<JPH::BodyID> own_bodies;
        own_bodies.reserve(jolt_items.size());
        for (const auto &item: jolt_items) own_bodies.push_back(item->get_body()->GetID());

        camera = std::make_shared<view::CollisionCamera>(
            canon_item,
            [&engine,
             own_bodies = std::move(own_bodies)](const glm::vec3 from, const glm::vec3 to) {
                return engine.ray_test(from, to, own_bodies);
            },
            wanted_frame_frequency);

        // the tank never collides with itself, like Bullet's pairwise
        // setIgnoreCollisionCheck
        static std::atomic<JPH::CollisionGroup::GroupID> next_group_id{0};
        const auto group_id = next_group_id++;

        const JPH::Ref group_filter =
            new JPH::GroupFilterTable(static_cast<JPH::uint>(jolt_items.size()));
        for (JPH::uint i = 0; i + 1 < jolt_items.size(); i++)
            for (JPH::uint j = i + 1; j < jolt_items.size(); j++)
                group_filter->DisableCollision(i, j);

        for (JPH::uint i = 0; i < jolt_items.size(); i++)
            jolt_items[i]->get_body()->SetCollisionGroup(
                JPH::CollisionGroup(group_filter, group_id, i));

        for (auto &item: jolt_items) item->get_body()->SetAllowSleeping(false);

        // register with engine
        for (const auto &item: jolt_items) engine.add_jolt_item(item);

        engine.add_jolt_item_producer([c = canon_item] { return c->produce_jolt_items(); });
    }

    std::shared_ptr<view::AbstractCamera> JoltTank::get_camera() const { return camera; }

    std::vector<std::shared_ptr<Item>> JoltTank::get_items() const { return items; }

    std::vector<std::shared_ptr<controller::Controller>> JoltTank::get_controllers() const {
        return controllers;
    }

    std::map<std::string, std::shared_ptr<Shape>> JoltTank::load_shell_shapes() const {
        return {{ShellItem::NAME, ShellItem::load_shape(file_reader)}};
    }

    bool JoltTank::is_dead() const {
        return std::ranges::any_of(life_items, [](const LifeItem *li) { return li->is_dead(); });
    }

    std::vector<ImpactInfo> JoltTank::consume_received_impacts() const {
        std::vector<ImpactInfo> impacts;
        for (const auto life_item: life_items) {
            auto item_impacts = life_item->consume_hits_received();
            impacts.insert(impacts.end(), item_impacts.begin(), item_impacts.end());
        }
        return impacts;
    }

    std::shared_ptr<Item> JoltTank::get_chassis() const { return chassis; }

    std::shared_ptr<Item> JoltTank::get_canon() const { return canon; }

    void JoltTank::kill_life_items() const {
        for (const auto life_item: life_items) life_item->kill();
    }

    void JoltTank::remove_constraints_from_engine() const {
        for (const auto &item: jolt_items) engine.remove_jolt_item_constraints(item);
    }

    JoltTank::~JoltTank() {
        controllers.clear();
        items.clear();
        jolt_items.clear();
        life_items.clear();
    }

}// namespace arenai::model
