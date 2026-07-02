//
// Created by samuel on 02/04/2023.
//

#include "./bullet_tank.h"

#include <algorithm>

#include "../bullet_engine.h"
#include "./parts/canon.h"
#include "./parts/chassis.h"
#include "./parts/shell.h"
#include "./parts/turret.h"
#include "./parts/wheel.h"

using namespace arenai;
using namespace arenai::model;
using namespace arenai::view;
using namespace arenai::controller;

namespace arenai::model {

    BulletTank::BulletTank(
        BulletPhysicEngine &engine, const std::shared_ptr<utils::AbstractFileReader> &file_reader,
        const std::string &tank_prefix_name, glm::vec3 chassis_pos,
        const float wanted_frame_frequency,
        const std::function<void(const ShellContactInfo &, Item *)> &on_contact_callback)
        : engine(engine), name(tank_prefix_name), camera(std::nullptr_t()),
          file_reader(file_reader) {

        glm::vec3 scale(0.5);

        // chassis
        constexpr float chassis_mass = 1e4f;
        auto chassis_item = std::make_shared<ChassisItem>(
            tank_prefix_name, file_reader, chassis_pos, scale, chassis_mass);

        chassis = chassis_item;
        items.push_back(chassis);
        bullet_items.push_back(chassis_item);
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
                tank_prefix_name + "_" + wheel_name, file_reader, wheel_pos + chassis_pos,
                wheel_pos, wheel_scale, wheel_mass, chassis_item->get_body(), front_axle_z,
                angle_factor);

            bullet_items.push_back(wheel);
            items.push_back(wheel);
            controllers.push_back(wheel);
            life_items.push_back(wheel.get());
        }

        std::vector<std::tuple<std::string, glm::vec3>> wheel_config{
            {"wheel_right_3", {-2.7, -1., -front_axle_z}},
            {"wheel_left_3", {2.7, -1., -front_axle_z}}};

        for (auto &[wheel_name, wheel_pos]: wheel_config) {
            auto wheel = std::make_shared<WheelItem>(
                tank_prefix_name + "_" + wheel_name, file_reader, wheel_pos + chassis_pos,
                wheel_pos, wheel_scale, wheel_mass, chassis_item->get_body(), front_axle_z);

            bullet_items.push_back(wheel);
            items.push_back(wheel);
            controllers.push_back(wheel);
            life_items.push_back(wheel.get());
        }

        // turret
        glm::vec3 turret_pos(0.f, 1.3f, 1.2f);
        glm::vec3 turret_scale(1.2f);
        auto turret = std::make_shared<TurretItem>(
            tank_prefix_name, file_reader, chassis_pos + turret_pos, turret_pos,
            scale * turret_scale, 300, chassis_item->get_body());
        bullet_items.push_back(turret);
        items.push_back(turret);
        controllers.push_back(turret);
        life_items.push_back(turret.get());

        // canon
        glm::vec3 canon_pos(0.f, 0.5f, 1.7f);
        glm::vec3 canon_scale = turret_scale;
        auto canon_item = std::make_shared<CanonItem>(
            tank_prefix_name, file_reader, chassis_pos + turret_pos + canon_pos, canon_pos,
            scale * canon_scale, 100, turret->get_body(), wanted_frame_frequency,
            [on_contact_callback](const glm::vec3 fire_pos, const glm::vec3 hit_pos, Item *item) {
                on_contact_callback({fire_pos, hit_pos}, item);
            });

        bullet_items.push_back(canon_item);
        items.push_back(canon_item);
        controllers.push_back(canon_item);
        life_items.push_back(canon_item.get());

        camera = canon_item;
        canon = canon_item;

        for (int i = 0; i < bullet_items.size() - 1; i++)
            for (int j = i + 1; j < bullet_items.size(); j++)
                bullet_items[i]->get_body()->setIgnoreCollisionCheck(
                    bullet_items[j]->get_body(), true);

        for (auto &item: bullet_items) item->get_body()->setActivationState(DISABLE_DEACTIVATION);

        // register with engine
        for (const auto &item: bullet_items) engine.add_bullet_item(item);

        engine.add_bullet_item_producer([c = canon_item]() { return c->produce_bullet_items(); });
    }

    std::shared_ptr<Camera> BulletTank::get_camera() { return camera; }

    std::vector<std::shared_ptr<Item>> BulletTank::get_items() { return items; }

    std::vector<std::shared_ptr<Controller>> BulletTank::get_controllers() { return controllers; }

    std::map<std::string, std::shared_ptr<Shape>> BulletTank::load_shell_shapes() const {
        return {{ShellItem::NAME, ShellItem::load_shape(file_reader)}};
    }

    bool BulletTank::is_dead() {
        return std::ranges::any_of(life_items, [](const LifeItem *li) { return li->is_dead(); });
    }

    std::shared_ptr<Item> BulletTank::get_chassis() { return chassis; }

    std::shared_ptr<Item> BulletTank::get_canon() { return canon; }

    void BulletTank::remove_constraints_from_engine() {
        for (const auto &item: bullet_items) engine.remove_bullet_item_constraints(item);
    }

    BulletTank::~BulletTank() {
        controllers.clear();
        items.clear();
        bullet_items.clear();
        life_items.clear();
    }

}// namespace arenai::model
