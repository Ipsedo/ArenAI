//
// Created by samuel on 28/09/2025.
//

#ifndef PHYVR_TANK_FACTORY_H
#define PHYVR_TANK_FACTORY_H

#include <map>
#include <memory>

#include <phyvr_controller/controller.h>
#include <phyvr_utils/file_reader.h>
#include <phyvr_view/camera.h>

#include "./item.h"

class TankFactory {
public:
    TankFactory(
        const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
        glm::vec3 chassis_pos);

    std::shared_ptr<Camera> get_camera();
    std::vector<std::shared_ptr<Item>> get_items();
    std::vector<std::shared_ptr<ItemProducer>> get_item_producers();
    std::vector<std::shared_ptr<Controller>> get_controllers();

    std::map<std::string, std::shared_ptr<Shape>> load_ammu_shapes() const;

    virtual bool is_dead();

    virtual ~TankFactory();

protected:
    virtual void on_fired_shell_contact(Item *item) = 0;

private:
    std::string name;

    std::shared_ptr<Camera> camera;
    std::vector<std::shared_ptr<Item>> items;
    std::vector<std::shared_ptr<ItemProducer>> item_producers;
    std::vector<std::shared_ptr<Controller>> controllers;

    std::shared_ptr<AbstractFileReader> file_reader;

    bool is_already_dead;
};

class EnemyTankFactory final : public TankFactory {
public:
    EnemyTankFactory(
        const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
        glm::vec3 chassis_pos, int max_frames_upside_down);
    float get_reward();

    bool is_dead() override;

    std::vector<std::shared_ptr<Item>> dead_and_get_items();

    std::vector<float> get_proprioception();

protected:
    void on_fired_shell_contact(Item *item) override;

private:
    float reward;
    int max_frames_upside_down;
    int curr_frame_upside_down;

    bool is_dead_already_triggered;
};

class PlayerTankFactory final : public TankFactory {
public:
    PlayerTankFactory(
        const std::shared_ptr<AbstractFileReader> &fileReader, const std::string &tankPrefixName,
        const glm::vec3 &chassisPos);

protected:
    void on_fired_shell_contact(Item *item) override;
};

#endif// PHYVR_TANK_FACTORY_H
