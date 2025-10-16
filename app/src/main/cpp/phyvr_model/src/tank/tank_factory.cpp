//
// Created by samuel on 02/04/2023.
//

#include <numeric>

#include <phyvr_model/tank_factory.h>

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
    float wheel_mass = 10.f;
    glm::vec3 wheel_scale = scale * glm::vec3(1.3, 1.1, 1.1);
    std::vector<std::tuple<std::string, glm::vec3>> front_wheel_config{
        {"dir_wheel_right_1", {-2.7, -1., 3.}}, {"dir_wheel_left_1", {2.7, -1., 3.}}};

    for (auto &[wheel_name, wheel_pos]: front_wheel_config) {
        auto wheel = std::make_shared<DirectionalWheelItem>(
            tank_prefix_name + "_" + wheel_name, file_reader, wheel_pos + chassis_pos, wheel_pos,
            wheel_scale, wheel_mass, chassis_item->get_body());

        items.push_back(wheel);
        controllers.push_back(wheel);
    }

    std::vector<std::tuple<std::string, glm::vec3>> wheel_config{
        {"wheel_right_2", {-2.7, -1., 0.}},
        {"wheel_left_2", {2.7, -1., 0.}},
        {"wheel_right_3", {-2.7, -1., -3.}},
        {"wheel_left_3", {2.7, -1., -3.}}};

    for (auto &[wheel_name, wheel_pos]: wheel_config) {
        auto wheel = std::make_shared<WheelItem>(
            tank_prefix_name + "_" + wheel_name, file_reader, wheel_pos + chassis_pos, wheel_pos,
            wheel_scale, wheel_mass, chassis_item->get_body());

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

std::map<std::string, std::shared_ptr<Shape>> TankFactory::load_ammu_shapes() const {
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

/*
 * Enemy
 */

EnemyTankFactory::EnemyTankFactory(
    const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
    const glm::vec3 chassis_pos, const int max_frames_upside_down)
    : TankFactory(file_reader, tank_prefix_name, chassis_pos), reward(0.f),
      max_frames_upside_down(max_frames_upside_down), curr_frame_upside_down(0),
      is_dead_already_triggered(false) {}

float EnemyTankFactory::get_reward() {
    float actual_reward = reward;

    // prepare next frame
    reward = 0.f;

    const auto chassis = get_items()[0];
    auto chassis_tr = chassis->get_body()->getWorldTransform();
    const btVector3 up(0.f, 1.f, 0.f);
    const btVector3 up_in_chassis = chassis_tr.getBasis() * up;

    if (const btScalar dot = up_in_chassis.normalized().dot(up.normalized()); dot < 0) {
        curr_frame_upside_down++;
        actual_reward -= 0.125f;
    } else curr_frame_upside_down = 0;

    if (is_dead()) actual_reward -= 1.f;

    return actual_reward;
}

void EnemyTankFactory::on_fired_shell_contact(Item *item) {
    bool self_shoot = false;
    for (const auto &i: get_items()) {
        if (i->get_name() == item->get_name()) {
            reward -= 0.25f;
            self_shoot = true;
        }
    }

    if (const auto &life_item = dynamic_cast<LifeItem *>(item); !self_shoot && life_item) {
        if (life_item->is_dead() && !life_item->is_already_dead()) reward += 1.0f;
        else if (!life_item->is_dead()) reward += 0.5f;
    }
}

bool EnemyTankFactory::is_dead() {
    return TankFactory::is_dead() || curr_frame_upside_down > max_frames_upside_down;
}

std::vector<std::shared_ptr<Item>> EnemyTankFactory::dead_and_get_items() {
    if (is_dead() && !is_dead_already_triggered) {
        is_dead_already_triggered = true;
        return get_items();
    }

    return {};
}

std::vector<float> EnemyTankFactory::get_proprioception() {
    const auto items = get_items();

    const auto &chassis = items[0];

    const auto chassis_pos = chassis->get_body()->getCenterOfMassPosition();
    const auto chassis_vel = chassis->get_body()->getLinearVelocity();

    const auto chassis_ang = chassis->get_body()->getOrientation();
    const auto chassis_ang_vel = chassis->get_body()->getAngularVelocity();

    std::vector result{chassis_pos.x(),    chassis_pos.y(),     chassis_pos.z(),
                       chassis_vel.x(),    chassis_vel.y(),     chassis_vel.z(),
                       chassis_ang.x(),    chassis_ang.y(),     chassis_ang.z(),
                       chassis_ang.w(),    chassis_ang_vel.x(), chassis_ang_vel.y(),
                       chassis_ang_vel.z()};
    result.reserve((3 * 2 + 4 + 3) * items.size());

    for (int i = 1; i < items.size(); i++) {
        const auto body = items[i]->get_body();

        auto pos = body->getCenterOfMassPosition() - chassis_pos;
        auto vel = body->getLinearVelocity();

        auto ang = body->getCenterOfMassTransform().getRotation();
        auto ang_vel = body->getAngularVelocity();

        result.insert(
            result.end(), {pos.x(), pos.y(), pos.z(), vel.x(), vel.y(), vel.y(), ang.x(), ang.y(),
                           ang.z(), ang.w(), ang_vel.x(), ang_vel.y(), ang_vel.z()});
    }
    return result;
}

/*
 * Player
 */

PlayerTankFactory::PlayerTankFactory(
    const std::shared_ptr<AbstractFileReader> &fileReader, const std::string &tankPrefixName,
    const glm::vec3 &chassisPos)
    : TankFactory(fileReader, tankPrefixName, chassisPos) {}

void PlayerTankFactory::on_fired_shell_contact(Item *item) {}
