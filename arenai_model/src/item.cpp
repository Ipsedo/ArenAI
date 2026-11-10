#include <utility>

#include <arenai_model/item.h>

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    Item::Item(std::string name) : name(std::move(name)), will_destroy(false) {}

    std::string Item::get_name() { return name; }

    bool Item::need_destroy() { return will_destroy; }

    void Item::destroy() { will_destroy = true; }

    void Item::on_contact(Item *other) {}

    void Item::tick() {}

    LifeItem::LifeItem(const float health_points)
        : initial_health_points(std::max(health_points, 0.f)), health_points(initial_health_points),
          already_dead(false) {}

    bool LifeItem::is_dead() const { return health_points <= 0.f; }

    float LifeItem::health_ratio() const {
        return initial_health_points > 0.f ? health_points / initial_health_points : 0.f;
    }

    float LifeItem::initial_health() const { return initial_health_points; }

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

    void LifeItem::restore() {
        health_points = initial_health_points;
        already_dead = false;
        hits_received.clear();
    }

    float LifeItem::receive_damages(const ImpactInfo &impact) {
        const float new_health_point = std::max(health_points - impact.damages, 0.f);
        const float received_damages = health_points - new_health_point;

        if (received_damages > 0.f) {
            auto received = impact;
            received.damages = received_damages;
            hits_received.push_back(received);
        }

        health_points = new_health_point;

        return received_damages;
    }

    std::vector<ImpactInfo> LifeItem::consume_hits_received() {
        return std::exchange(hits_received, {});
    }

}// namespace arenai::model
