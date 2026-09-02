//
// Created by samuel on 19/03/2023.
//

#include <utility>

#include <arenai_model/item.h>

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    /*
     * Base Item
     */

    Item::Item(std::string name) : name(std::move(name)), will_destroy(false) {}

    std::string Item::get_name() { return name; }

    bool Item::need_destroy() { return will_destroy; }

    void Item::destroy() { will_destroy = true; }

    void Item::on_contact(Item *other) {}

    void Item::tick() {}

    /*
     * Life Item
     */

    LifeItem::LifeItem(const float health_points)
        : health_points(std::max(health_points, 0.f)), already_dead(false) {}

    bool LifeItem::is_dead() const { return health_points <= 0.f; }

    bool LifeItem::is_already_dead() {
        if (is_dead() && !already_dead) {
            already_dead = true;
            return false;
        }
        return already_dead;
    }

    void LifeItem::kill() {
        health_points = 0.f;
        already_dead = true;
    }

    float LifeItem::receive_damages(const ImpactInfo &impact) {
        const float new_health_point = std::max(health_points - impact.damages, 0.f);
        const float received_damages = health_points - new_health_point;

        if (received_damages > 0.f) hits_received.push_back(impact);

        health_points = new_health_point;

        return received_damages;
    }

    std::vector<ImpactInfo> LifeItem::consume_hits_received() {
        return std::exchange(hits_received, {});
    }

}// namespace arenai::model
