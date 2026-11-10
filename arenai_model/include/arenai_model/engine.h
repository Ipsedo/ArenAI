#ifndef ARENAI_ENGINE_H
#define ARENAI_ENGINE_H

#include <memory>
#include <optional>
#include <vector>

#include <glm/glm.hpp>

#include "./item.h"

namespace arenai::model {

    class ItemFactory;
    class TankFactory;

    class AbstractPhysicEngine {
    public:
        virtual ~AbstractPhysicEngine() = default;

        virtual void step(float delta) = 0;

        virtual std::optional<glm::vec3> ray_cast(glm::vec3 from, glm::vec3 to) const = 0;

        virtual std::vector<std::shared_ptr<Item>> get_items() = 0;

        virtual void remove_bodies_and_constraints() = 0;

        virtual std::shared_ptr<ItemFactory> get_item_factory() = 0;
        virtual std::shared_ptr<TankFactory> get_tank_factory() = 0;
    };

    std::unique_ptr<AbstractPhysicEngine> make_physic_engine(float wanted_frequency);

}// namespace arenai::model

#endif// ARENAI_ENGINE_H
