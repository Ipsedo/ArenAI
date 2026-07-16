//
// Created by samuel on 15/07/2026.
//

#ifndef ARENAI_POST_PROCESSING_COMPOSITE_H
#define ARENAI_POST_PROCESSING_COMPOSITE_H

#include "./effect.h"

namespace arenai::view {

    // final composite (ACES tonemapping, color grading, chromatic aberration,
    // vignette, film grain/dithering) onto the default framebuffer; consumes
    // "ao", "bloom", "rays" and "ray_strength", so it must run last
    class CompositeEffect : public AbstractPostProcessingEffect {
    public:
        CompositeEffect(int width, int height);

        void render(FrameContext &context) override;
    };

}// namespace arenai::view

#endif// ARENAI_POST_PROCESSING_COMPOSITE_H
