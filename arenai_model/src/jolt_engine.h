//
// Created by claude on 30/07/2026.
//

#ifndef ARENAI_JOLT_ENGINE_H
#define ARENAI_JOLT_ENGINE_H

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <vector>

#include <glm/glm.hpp>

// Jolt.h defines the platform macros every other Jolt header relies on: it has
// to stay first, out of clang-format's alphabetical reach
// clang-format off
#include <Jolt/Jolt.h>
// clang-format on
#include <Jolt/Core/JobSystemSingleThreaded.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/PhysicsSystem.h>

#include <arenai_model/engine.h>
#include <arenai_model/item_factory.h>

#include "./jolt_item.h"

namespace arenai::model {

    // object layers: static bodies never collide together, everything else does
    namespace layers {
        constexpr JPH::ObjectLayer NON_MOVING = 0;
        constexpr JPH::ObjectLayer MOVING = 1;
        constexpr JPH::uint NUM_LAYERS = 2;
    }// namespace layers

    class JoltPhysicEngine final : public AbstractPhysicEngine {
    public:
        explicit JoltPhysicEngine(float wanted_frequency);

        void step(float delta) override;

        std::vector<std::shared_ptr<Item>> get_items() override;

        void remove_bodies_and_constraints() override;

        std::shared_ptr<ItemFactory> get_item_factory() override;
        std::shared_ptr<TankFactory> get_tank_factory() override;

        ~JoltPhysicEngine() override;

        void add_jolt_item(const std::shared_ptr<JoltItem> &item);
        void
        add_jolt_item_producer(std::function<std::vector<std::shared_ptr<JoltItem>>()> producer);
        void remove_jolt_item_constraints(const std::shared_ptr<JoltItem> &item);

        // Fraction of the [from -> to] segment at which the closest non-excluded
        // body is hit, std::nullopt when the path is free. Read-only query: safe
        // to call concurrently as long as the world is not stepping.
        std::optional<float>
        ray_test(glm::vec3 from, glm::vec3 to, const std::vector<JPH::BodyID> &excluded) const;

        // for items building their body / reading it back
        JPH::BodyInterface &get_body_interface();
        float get_wanted_frequency() const;

        // Bullet's motion-state latency interpolation offset: leftover time
        // minus one fixed timestep, always in [-wanted_frequency, 0)
        float get_interpolation_delta() const;

    private:
        // every step's penetrating contact points, kept until the next simulated
        // sub-step so that a step() call that simulates nothing (accumulated time
        // under the fixed timestep) re-fires the same contacts, like Bullet's
        // persistent manifolds did
        class BufferedContactListener final : public JPH::ContactListener {
        public:
            void OnContactAdded(
                const JPH::Body &body1, const JPH::Body &body2,
                const JPH::ContactManifold &manifold, JPH::ContactSettings &settings) override;
            void OnContactPersisted(
                const JPH::Body &body1, const JPH::Body &body2,
                const JPH::ContactManifold &manifold, JPH::ContactSettings &settings) override;

            void clear();
            std::vector<std::pair<Item *, Item *>> snapshot();
            void forget(const Item *item);

        private:
            void record(
                const JPH::Body &body1, const JPH::Body &body2,
                const JPH::ContactManifold &manifold);

            std::mutex contacts_mutex;
            std::vector<std::pair<Item *, Item *>> contacts;
        };

        std::shared_mutex items_mutex;

        float wanted_frequency;
        // Bullet's stepSimulation leftover time: accumulates deltas and only
        // simulates whole fixed sub-steps, one at most per step() call
        float local_time;

        std::unique_ptr<JPH::TempAllocatorImpl> temp_allocator;
        std::unique_ptr<JPH::JobSystemSingleThreaded> job_system;

        std::unique_ptr<JPH::BroadPhaseLayerInterface> broad_phase_layer_interface;
        std::unique_ptr<JPH::ObjectVsBroadPhaseLayerFilter> object_vs_broad_phase_filter;
        std::unique_ptr<JPH::ObjectLayerPairFilter> object_layer_pair_filter;

        std::unique_ptr<JPH::PhysicsSystem> physics_system;
        BufferedContactListener contact_listener;

        std::vector<std::shared_ptr<JoltItem>> items;
        std::vector<std::function<std::vector<std::shared_ptr<JoltItem>>()>> jolt_item_producers;

        // constraints referencing a body, removed with it like Bullet's
        // btRigidBody constraint refs
        std::map<JPH::BodyID, std::vector<JPH::Ref<JPH::TwoBodyConstraint>>> constraints_per_body;

        std::shared_ptr<ItemFactory> item_factory;
        std::shared_ptr<TankFactory> tank_factory;

        void add_item_locked(const std::shared_ptr<JoltItem> &item);
        void remove_dead_items();
    };

}// namespace arenai::model

#endif// ARENAI_JOLT_ENGINE_H
