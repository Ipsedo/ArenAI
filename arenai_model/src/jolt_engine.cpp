//
// Created by claude on 30/07/2026.
//

#include "./jolt_engine.h"

#include <algorithm>
#include <cmath>

#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/NarrowPhaseQuery.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/RegisterTypes.h>

#include "./items/jolt_item_factory.h"
#include "./tank/jolt_tank_factory.h"

using namespace arenai;
using namespace arenai::model;

namespace {

    // process-wide Jolt runtime, initialized once whatever the number of engines
    void ensure_jolt_initialized() {
        static std::once_flag once;
        std::call_once(once, [] {
            JPH::RegisterDefaultAllocator();
            JPH::Factory::sInstance = new JPH::Factory();
            JPH::RegisterTypes();
        });
    }

    namespace broad_phase_layers {
        constexpr JPH::BroadPhaseLayer NON_MOVING(0);
        constexpr JPH::BroadPhaseLayer MOVING(1);
        constexpr JPH::uint NUM_LAYERS = 2;
    }// namespace broad_phase_layers

    class BroadPhaseLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface {
    public:
        JPH::uint GetNumBroadPhaseLayers() const override { return broad_phase_layers::NUM_LAYERS; }

        JPH::BroadPhaseLayer GetBroadPhaseLayer(const JPH::ObjectLayer layer) const override {
            return layer == arenai::model::layers::NON_MOVING ? broad_phase_layers::NON_MOVING
                                                              : broad_phase_layers::MOVING;
        }

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
        const char *GetBroadPhaseLayerName(const JPH::BroadPhaseLayer layer) const override {
            return layer == broad_phase_layers::NON_MOVING ? "NON_MOVING" : "MOVING";
        }
#endif
    };

    class ObjectVsBroadPhaseLayerFilterImpl final : public JPH::ObjectVsBroadPhaseLayerFilter {
    public:
        bool ShouldCollide(
            const JPH::ObjectLayer layer1, const JPH::BroadPhaseLayer layer2) const override {
            return layer1 != arenai::model::layers::NON_MOVING
                   || layer2 != broad_phase_layers::NON_MOVING;
        }
    };

    class ObjectLayerPairFilterImpl final : public JPH::ObjectLayerPairFilter {
    public:
        bool
        ShouldCollide(const JPH::ObjectLayer layer1, const JPH::ObjectLayer layer2) const override {
            return layer1 != arenai::model::layers::NON_MOVING
                   || layer2 != arenai::model::layers::NON_MOVING;
        }
    };

}// namespace

namespace arenai::model {

    void JoltPhysicEngine::BufferedContactListener::record(
        const JPH::Body &body1, const JPH::Body &body2, const JPH::ContactManifold &manifold) {
        // Bullet fired on_contact once per penetrating manifold point
        if (manifold.mPenetrationDepth <= 0.f) return;

        const auto item_a = reinterpret_cast<Item *>(body1.GetUserData());
        const auto item_b = reinterpret_cast<Item *>(body2.GetUserData());
        if (item_a == nullptr || item_b == nullptr) return;

        std::lock_guard lock(contacts_mutex);
        for (JPH::uint i = 0; i < manifold.mRelativeContactPointsOn1.size(); i++)
            contacts.emplace_back(item_a, item_b);
    }

    void JoltPhysicEngine::BufferedContactListener::OnContactAdded(
        const JPH::Body &body1, const JPH::Body &body2, const JPH::ContactManifold &manifold,
        JPH::ContactSettings &settings) {
        record(body1, body2, manifold);
    }

    void JoltPhysicEngine::BufferedContactListener::OnContactPersisted(
        const JPH::Body &body1, const JPH::Body &body2, const JPH::ContactManifold &manifold,
        JPH::ContactSettings &settings) {
        record(body1, body2, manifold);
    }

    void JoltPhysicEngine::BufferedContactListener::clear() {
        std::lock_guard lock(contacts_mutex);
        contacts.clear();
    }

    std::vector<std::pair<Item *, Item *>> JoltPhysicEngine::BufferedContactListener::snapshot() {
        std::lock_guard lock(contacts_mutex);
        return contacts;
    }

    void JoltPhysicEngine::BufferedContactListener::forget(const Item *item) {
        std::lock_guard lock(contacts_mutex);
        std::erase_if(contacts, [item](const auto &pair) {
            return pair.first == item || pair.second == item;
        });
    }

    JoltPhysicEngine::JoltPhysicEngine(const float wanted_frequency)
        : wanted_frequency(wanted_frequency), local_time(0.f) {
        ensure_jolt_initialized();

        temp_allocator = std::make_unique<JPH::TempAllocatorImpl>(32 * 1024 * 1024);
        job_system = std::make_unique<JPH::JobSystemSingleThreaded>(JPH::cMaxPhysicsJobs);

        broad_phase_layer_interface = std::make_unique<BroadPhaseLayerInterfaceImpl>();
        object_vs_broad_phase_filter = std::make_unique<ObjectVsBroadPhaseLayerFilterImpl>();
        object_layer_pair_filter = std::make_unique<ObjectLayerPairFilterImpl>();

        physics_system = std::make_unique<JPH::PhysicsSystem>();
        physics_system->Init(
            10240, 0, 65536, 10240, *broad_phase_layer_interface, *object_vs_broad_phase_filter,
            *object_layer_pair_filter);

        physics_system->SetGravity(JPH::Vec3(0.f, -9.8f, 0.f));
        physics_system->SetContactListener(&contact_listener);

        // Bullet combines friction and restitution by multiplying them (Jolt
        // defaults to geometric mean / max): keep the same effective traction
        physics_system->SetCombineFriction(
            [](const JPH::Body &body1, const JPH::SubShapeID &, const JPH::Body &body2,
               const JPH::SubShapeID &) { return body1.GetFriction() * body2.GetFriction(); });
        physics_system->SetCombineRestitution([](const JPH::Body &body1, const JPH::SubShapeID &,
                                                 const JPH::Body &body2, const JPH::SubShapeID &) {
            return body1.GetRestitution() * body2.GetRestitution();
        });

        item_factory = std::make_shared<JoltItemFactory>(*this);
        tank_factory = std::make_shared<JoltTankFactory>(*this, wanted_frequency);
    }

    void JoltPhysicEngine::add_item_locked(const std::shared_ptr<JoltItem> &item) {
        items.push_back(item);

        auto *body = item->get_body();
        physics_system->GetBodyInterface().AddBody(body->GetID(), JPH::EActivation::Activate);

        for (const auto &constraint: item->get_constraints()) {
            physics_system->AddConstraint(constraint);
            constraints_per_body[constraint->GetBody1()->GetID()].push_back(constraint);
            constraints_per_body[constraint->GetBody2()->GetID()].push_back(constraint);
        }
    }

    void JoltPhysicEngine::add_jolt_item(const std::shared_ptr<JoltItem> &item) {
        std::unique_lock lock(items_mutex);
        add_item_locked(item);
    }

    void JoltPhysicEngine::add_jolt_item_producer(
        std::function<std::vector<std::shared_ptr<JoltItem>>()> producer) {
        std::unique_lock lock(items_mutex);
        jolt_item_producers.push_back(std::move(producer));
    }

    void JoltPhysicEngine::remove_jolt_item_constraints(const std::shared_ptr<JoltItem> &item) {
        std::unique_lock lock(items_mutex);
        for (const auto &constraint: item->get_constraints()) {
            physics_system->RemoveConstraint(constraint);

            for (auto *body: {constraint->GetBody1(), constraint->GetBody2()}) {
                if (const auto it = constraints_per_body.find(body->GetID());
                    it != constraints_per_body.end()) {
                    std::erase(it->second, constraint);
                    if (it->second.empty()) constraints_per_body.erase(it);
                }
            }
        }
    }

    void JoltPhysicEngine::remove_dead_items() {
        std::unique_lock lock(items_mutex);

        for (const auto &item: items) item->tick();

        auto &body_interface = physics_system->GetBodyInterface();

        for (int i = static_cast<int>(items.size()) - 1; i >= 0; i--) {
            if (const auto item = items[i]; item->need_destroy()) {
                const auto body_id = item->get_body()->GetID();

                // like Bullet, removing a body also drops every constraint
                // still referencing it
                if (const auto it = constraints_per_body.find(body_id);
                    it != constraints_per_body.end()) {
                    for (const auto &constraint: it->second) {
                        physics_system->RemoveConstraint(constraint);

                        const auto other_id = constraint->GetBody1()->GetID() == body_id
                                                  ? constraint->GetBody2()->GetID()
                                                  : constraint->GetBody1()->GetID();
                        if (const auto other_it = constraints_per_body.find(other_id);
                            other_it != constraints_per_body.end()) {
                            std::erase(other_it->second, constraint);
                            if (other_it->second.empty()) constraints_per_body.erase(other_it);
                        }
                    }
                    constraints_per_body.erase(body_id);
                }

                contact_listener.forget(item.get());

                body_interface.RemoveBody(body_id);
                body_interface.DestroyBody(body_id);

                items.erase(items.begin() + i);
            }
        }
    }

    void JoltPhysicEngine::step(const float delta) {
        {
            std::unique_lock lock(items_mutex);
            for (const auto &producer: jolt_item_producers)
                for (const auto &item: producer()) add_item_locked(item);
        }

        // Bullet's stepSimulation(delta, 1, wanted_frequency): accumulate time,
        // simulate at most one fixed sub-step per call, drop the surplus
        local_time += delta;
        if (local_time >= wanted_frequency) {
            local_time -= std::floor(local_time / wanted_frequency) * wanted_frequency;

            contact_listener.clear();
            physics_system->Update(wanted_frequency, 1, temp_allocator.get(), job_system.get());
        }

        for (const auto &[item_a, item_b]: contact_listener.snapshot()) {
            item_a->on_contact(item_b);
            item_b->on_contact(item_a);
        }

        remove_dead_items();
    }

    std::optional<float> JoltPhysicEngine::ray_test(
        const glm::vec3 from, const glm::vec3 to, const std::vector<JPH::BodyID> &excluded) const {

        class ExcludingBodyFilter final : public JPH::BodyFilter {
        public:
            explicit ExcludingBodyFilter(const std::vector<JPH::BodyID> &excluded)
                : excluded(excluded) {}

            bool ShouldCollide(const JPH::BodyID &body_id) const override {
                return std::ranges::find(excluded, body_id) == excluded.end();
            }

        private:
            const std::vector<JPH::BodyID> &excluded;
        };

        const JPH::RVec3 origin(from.x, from.y, from.z);
        const JPH::Vec3 direction(to.x - from.x, to.y - from.y, to.z - from.z);

        const JPH::RRayCast ray(origin, direction);
        JPH::RayCastResult result;

        const ExcludingBodyFilter body_filter(excluded);

        if (physics_system->GetNarrowPhaseQuery().CastRay(ray, result, {}, {}, body_filter))
            return result.mFraction;
        return std::nullopt;
    }

    std::vector<std::shared_ptr<Item>> JoltPhysicEngine::get_items() {
        std::shared_lock lock(items_mutex);
        return {items.begin(), items.end()};
    }

    std::shared_ptr<ItemFactory> JoltPhysicEngine::get_item_factory() { return item_factory; }

    std::shared_ptr<TankFactory> JoltPhysicEngine::get_tank_factory() { return tank_factory; }

    JPH::BodyInterface &JoltPhysicEngine::get_body_interface() {
        return physics_system->GetBodyInterface();
    }

    float JoltPhysicEngine::get_wanted_frequency() const { return wanted_frequency; }

    float JoltPhysicEngine::get_interpolation_delta() const {
        return local_time - wanted_frequency;
    }

    void JoltPhysicEngine::remove_bodies_and_constraints() {
        std::unique_lock lock(items_mutex);

        for (const auto &[body_id, constraints]: constraints_per_body)
            for (const auto &constraint: constraints)
                if (constraint->GetBody1()->GetID() == body_id)// remove each constraint once
                    physics_system->RemoveConstraint(constraint);
        constraints_per_body.clear();

        auto &body_interface = physics_system->GetBodyInterface();
        for (const auto &item: items) {
            const auto body_id = item->get_body()->GetID();
            body_interface.RemoveBody(body_id);
            body_interface.DestroyBody(body_id);
        }

        contact_listener.clear();
        jolt_item_producers.clear();
        items.clear();
    }

    JoltPhysicEngine::~JoltPhysicEngine() { remove_bodies_and_constraints(); }

}// namespace arenai::model
