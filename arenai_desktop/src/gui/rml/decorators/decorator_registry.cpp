//
// Created by samuel on 25/10/2026.
//

#include "./decorator_registry.h"

#include "./corner_reticle.h"
#include "./damage_arc.h"
#include "./facets.h"
#include "./hit_marker.h"
#include "./reticle.h"
#include "./slant.h"

namespace arenai::desktop::gui {

    void DecoratorRegistry::add(
        const Rml::String &name, std::unique_ptr<Rml::DecoratorInstancer> instancer) {
        Rml::Factory::RegisterDecoratorInstancer(name, instancer.get());
        instancers_.push_back(std::move(instancer));
    }

    void register_menu_decorators(DecoratorRegistry &registry) {
        registry.add("corner-reticle", std::make_unique<CornerReticleDecoratorInstancer>());
        registry.add("slant", std::make_unique<SlantDecoratorInstancer>());
        registry.add("facets", std::make_unique<FacetsDecoratorInstancer>());
        registry.add("reticle", std::make_unique<ReticleDecoratorInstancer>());
        registry.add("hit-marker", std::make_unique<HitMarkerDecoratorInstancer>());
        registry.add("damage-arc", std::make_unique<DamageArcDecoratorInstancer>());
    }

}// namespace arenai::desktop::gui
