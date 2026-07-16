//
// Created by samuel on 15/07/2026.
//

#ifndef ARENAI_POST_PROCESSING_GOD_RAYS_H
#define ARENAI_POST_PROCESSING_GOD_RAYS_H

#include "./effect.h"

namespace arenai::view {

    // volumetric sun rays: projects the sun onto the screen and fades the
    // effect out as it faces away or leaves the frame; publishes "rays" and
    // the "ray_strength" scalar (0 skips the draw, the composite then ignores
    // the stale texture)
    class GodRaysEffect : public AbstractPostProcessingEffect {
    public:
        GodRaysEffect(int width, int height);

        void render(FrameContext &context) override;

    private:
        static constexpr float RAY_STRENGTH = 0.4f;
    };

}// namespace arenai::view

#endif// ARENAI_POST_PROCESSING_GOD_RAYS_H
