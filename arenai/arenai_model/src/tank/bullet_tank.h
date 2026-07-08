//
// Created by samuel on 02/04/2023.
//

#ifndef ARENAI_BULLET_TANK_H
#define ARENAI_BULLET_TANK_H

#include <functional>

#include <arenai_model/tank.h>

#include "../bullet_item.h"

namespace arenai::model {

    class BulletPhysicEngine;
    class CanonItem;

    class BulletTank : virtual public Tank {
    public:
        BulletTank(
            BulletPhysicEngine &engine,
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            const std::string &tank_prefix_name, glm::vec3 chassis_pos,
            float wanted_frame_frequency,
            const std::function<void(const ShellContactInfo &, Item *)> &on_contact_callback);

        std::shared_ptr<view::AbstractCamera> get_camera() override;
        std::vector<std::shared_ptr<Item>> get_items() override;
        std::vector<std::shared_ptr<controller::Controller>> get_controllers() override;
        std::map<std::string, std::shared_ptr<Shape>> load_shell_shapes() const override;
        bool is_dead() override;
        std::shared_ptr<Item> get_chassis() override;
        std::shared_ptr<Item> get_canon() override;

        ~BulletTank() override;

    protected:
        void remove_constraints_from_engine();
        BulletPhysicEngine &engine;

    private:
        std::string name;
        std::shared_ptr<view::AbstractCamera> camera;
        std::vector<std::shared_ptr<Item>> items;
        std::vector<std::shared_ptr<BulletItem>> bullet_items;
        std::vector<std::shared_ptr<controller::Controller>> controllers;
        std::vector<LifeItem *> life_items;
        std::shared_ptr<Item> chassis;
        std::shared_ptr<Item> canon;
        std::shared_ptr<utils::AbstractFileReader> file_reader;
    };

}// namespace arenai::model

#endif// ARENAI_BULLET_TANK_H
