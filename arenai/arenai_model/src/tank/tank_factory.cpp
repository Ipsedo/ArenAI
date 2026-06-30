//
// Created by samuel on 02/04/2023.
//

#include <algorithm>
#include <numeric>

#include <arenai_model/tank_factory.h>

#include "bullet_item.h"
#include "canon.h"
#include "chassis.h"
#include "shell.h"
#include "turret.h"
#include "wheel.h"

TankFactory::TankFactory(
    const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
    glm::vec3 chassis_pos, float wanted_frame_frequency)
    : name(tank_prefix_name), camera(std::nullptr_t()), file_reader(file_reader) {

    glm::vec3 scale(0.5);

    // chassis
    constexpr float chassis_mass = 1e4f;
    auto chassis_item = std::make_shared<ChassisItem>(
        tank_prefix_name, file_reader, chassis_pos, scale, chassis_mass);

    chassis = chassis_item;
    items.push_back(chassis);

    // wheels
    constexpr float front_axle_z = 3.f;

    constexpr float wheel_mass = 150.f;
    glm::vec3 wheel_scale = scale * glm::vec3(1.3, 1.1, 1.1);

    std::vector<std::shared_ptr<BulletItem>> bullet_items;
    bullet_items.push_back(chassis_item);

    std::vector<std::tuple<std::string, glm::vec3, float>> front_wheel_config{
        {"wheel_right_1", {-2.7, -1., front_axle_z}, 1.f},
        {"wheel_left_1", {2.7, -1., front_axle_z}, 1.f},
        {"wheel_right_2", {-2.7, -1., 0.}, 0.5f},
        {"wheel_left_2", {2.7, -1., 0.}, 0.5f}};

    for (auto &[wheel_name, wheel_pos, angle_factor]: front_wheel_config) {
        auto wheel = std::make_shared<DirectionalWheelItem>(
            tank_prefix_name + "_" + wheel_name, file_reader, wheel_pos + chassis_pos, wheel_pos,
            wheel_scale, wheel_mass, chassis_item->get_body(), front_axle_z, angle_factor);

        bullet_items.push_back(wheel);
        items.push_back(wheel);
        controllers.push_back(wheel);
    }

    std::vector<std::tuple<std::string, glm::vec3>> wheel_config{
        {"wheel_right_3", {-2.7, -1., -front_axle_z}}, {"wheel_left_3", {2.7, -1., -front_axle_z}}};

    for (auto &[wheel_name, wheel_pos]: wheel_config) {
        auto wheel = std::make_shared<WheelItem>(
            tank_prefix_name + "_" + wheel_name, file_reader, wheel_pos + chassis_pos, wheel_pos,
            wheel_scale, wheel_mass, chassis_item->get_body(), front_axle_z);

        bullet_items.push_back(wheel);
        items.push_back(wheel);
        controllers.push_back(wheel);
    }

    // turret
    glm::vec3 turret_pos(0.f, 1.3f, 1.2f);
    glm::vec3 turret_scale(1.2f);
    auto turret = std::make_shared<TurretItem>(
        tank_prefix_name, file_reader, chassis_pos + turret_pos, turret_pos, scale * turret_scale,
        300, chassis_item->get_body());
    bullet_items.push_back(turret);
    items.push_back(turret), controllers.push_back(turret);

    // canon
    glm::vec3 canon_pos(0.f, 0.5f, 1.7f);
    glm::vec3 canon_scale = turret_scale;
    const auto canon_item = std::make_shared<CanonItem>(
        tank_prefix_name, file_reader, chassis_pos + turret_pos + canon_pos, canon_pos,
        scale * canon_scale, 100, turret->get_body(), wanted_frame_frequency,
        [this](glm::vec3 fire_pos, glm::vec3 current_pos, Item *i) {
            on_fired_shell_contact({fire_pos, current_pos}, i);
        });

    bullet_items.push_back(canon_item);
    item_producers.push_back(canon_item);
    items.push_back(canon_item), controllers.push_back(canon_item);

    camera = canon_item;
    canon = canon_item;

    for (int i = 0; i < bullet_items.size() - 1; i++)
        for (int j = i + 1; j < bullet_items.size(); j++)
            bullet_items[i]->get_body()->setIgnoreCollisionCheck(bullet_items[j]->get_body(), true);

    for (auto &item: bullet_items) item->get_body()->setActivationState(DISABLE_DEACTIVATION);
}

std::shared_ptr<Camera> TankFactory::get_camera() { return camera; }

std::vector<std::shared_ptr<Item>> TankFactory::get_items() { return items; }

std::vector<std::shared_ptr<Controller>> TankFactory::get_controllers() { return controllers; }

std::map<std::string, std::shared_ptr<Shape>> TankFactory::load_shell_shapes() const {
    return {{ShellItem::NAME, ShellItem::load_shape(file_reader)}};
}

std::vector<std::shared_ptr<ItemProducer>> TankFactory::get_item_producers() {
    return item_producers;
}

bool TankFactory::is_dead() {
    return std::ranges::any_of(items, [](const std::shared_ptr<Item> &item) {
        if (const auto life_item = dynamic_cast<LifeItem *>(item.get()))
            return life_item->is_dead();
        return false;
    });
}

std::shared_ptr<Item> TankFactory::get_chassis() { return chassis; }

std::shared_ptr<Item> TankFactory::get_canon() { return canon; }

TankFactory::~TankFactory() {
    item_producers.clear();
    controllers.clear();
    items.clear();
}
