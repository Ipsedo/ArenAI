//
// Created by samuel on 28/09/2025.
//

#ifndef ARENAI_TANK_FACTORY_H
#define ARENAI_TANK_FACTORY_H

#include <map>
#include <memory>

#include <arenai_controller/controller.h>
#include <arenai_utils/file_reader.h>
#include <arenai_view/camera.h>

#include "./item.h"
#include "src/tank/canon.h"
#include "src/tank/chassis.h"

class TankFactory {
public:
    TankFactory(
        const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
        glm::vec3 chassis_pos, float wanted_frame_frequency);

    std::shared_ptr<Camera> get_camera();
    std::vector<std::shared_ptr<Item>> get_items();
    std::vector<std::shared_ptr<ItemProducer>> get_item_producers();
    std::vector<std::shared_ptr<Controller>> get_controllers();

    std::map<std::string, std::shared_ptr<Shape>> load_shell_shapes() const;

    virtual bool is_dead();

    virtual ~TankFactory();

protected:
    virtual void on_fired_shell_contact(Item *item) = 0;

    std::shared_ptr<ChassisItem> get_chassis();
    std::shared_ptr<CanonItem> get_canon();

private:
    std::string name;

    std::shared_ptr<Camera> camera;
    std::vector<std::shared_ptr<Item>> items;
    std::vector<std::shared_ptr<ItemProducer>> item_producers;
    std::vector<std::shared_ptr<Controller>> controllers;

    std::shared_ptr<ChassisItem> chassis;
    std::shared_ptr<CanonItem> canon;

    std::shared_ptr<AbstractFileReader> file_reader;
};

#endif// ARENAI_TANK_FACTORY_H
