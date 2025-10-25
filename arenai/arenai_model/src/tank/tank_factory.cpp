//
// Created by samuel on 02/04/2023.
//

#include <numeric>

#include <arenai_model/tank_factory.h>

#include "./canon.h"
#include "./chassis.h"
#include "./shell.h"
#include "./turret.h"
#include "./wheel.h"

TankFactory::TankFactory(
    const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
    glm::vec3 chassis_pos)
    : name(tank_prefix_name), camera(std::nullptr_t()), items(), item_producers(), controllers(),
      file_reader(file_reader) {

    glm::vec3 scale(0.5);

    // chassis
    auto chassis_item =
        std::make_shared<ChassisItem>(tank_prefix_name, file_reader, chassis_pos, scale, 2000.f);

    items.push_back(chassis_item);

    // wheels
    float front_axle_z = 3.f;

    float wheel_mass = 10.f;
    glm::vec3 wheel_scale = scale * glm::vec3(1.3, 1.1, 1.1);
    std::vector<std::tuple<std::string, glm::vec3>> front_wheel_config{
        {"dir_wheel_right_1", {-2.7, -1., front_axle_z}},
        {"dir_wheel_left_1", {2.7, -1., front_axle_z}}};

    for (auto &[wheel_name, wheel_pos]: front_wheel_config) {
        auto wheel = std::make_shared<DirectionalWheelItem>(
            tank_prefix_name + "_" + wheel_name, file_reader, wheel_pos + chassis_pos, wheel_pos,
            wheel_scale, wheel_mass, chassis_item->get_body(), front_axle_z);

        items.push_back(wheel);
        controllers.push_back(wheel);
    }

    std::vector<std::tuple<std::string, glm::vec3>> wheel_config{
        {"wheel_right_2", {-2.7, -1., 0.}},
        {"wheel_left_2", {2.7, -1., 0.}},
        {"wheel_right_3", {-2.7, -1., -front_axle_z}},
        {"wheel_left_3", {2.7, -1., -front_axle_z}}};

    for (auto &[wheel_name, wheel_pos]: wheel_config) {
        auto wheel = std::make_shared<WheelItem>(
            tank_prefix_name + "_" + wheel_name, file_reader, wheel_pos + chassis_pos, wheel_pos,
            wheel_scale, wheel_mass, chassis_item->get_body(), front_axle_z);

        items.push_back(wheel);
        controllers.push_back(wheel);
    }

    // turret
    glm::vec3 turret_pos(0.f, 1.3f, 1.2f);
    glm::vec3 turret_scale(1.2f);
    auto turret = std::make_shared<TurretItem>(
        tank_prefix_name, file_reader, chassis_pos + turret_pos, turret_pos, scale * turret_scale,
        200, chassis_item->get_body());
    items.push_back(turret), controllers.push_back(turret);

    // canon
    glm::vec3 canon_pos(0.f, 0.5f, 1.7f);
    glm::vec3 canon_scale = turret_scale;
    auto canon = std::make_shared<CanonItem>(
        tank_prefix_name, file_reader, chassis_pos + turret_pos + canon_pos, canon_pos,
        scale * canon_scale, 50, turret->get_body(),
        [this](Item *i) { on_fired_shell_contact(i); });

    item_producers.push_back(canon);
    items.push_back(canon), controllers.push_back(canon);
    camera = canon;

    for (int i = 0; i < items.size() - 1; i++)
        for (int j = i + 1; j < items.size(); j++)
            items[i]->get_body()->setIgnoreCollisionCheck(items[j]->get_body(), true);

    for (auto &item: items) item->get_body()->setActivationState(DISABLE_DEACTIVATION);
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
    for (const auto &item: items)
        if (const auto life_item = std::dynamic_pointer_cast<LifeItem>(item); life_item->is_dead())
            return true;
    return false;
}

TankFactory::~TankFactory() {
    item_producers.clear();
    controllers.clear();
    items.clear();
}
