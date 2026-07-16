//
// Created by samuel on 15/07/2026.
//

#ifndef ARENAI_POST_PROCESSING_SSAO_H
#define ARENAI_POST_PROCESSING_SSAO_H

#include "./effect.h"

namespace arenai::view {

    // half-res ambient occlusion from the depth buffer, publishes "ao_raw"
    // (blurred by AoBlurEffect before being consumed)
    class SsaoEffect : public AbstractPostProcessingEffect {
    public:
        SsaoEffect(int width, int height);

        void render(FrameContext &context) override;
    };

}// namespace arenai::view

#endif// ARENAI_POST_PROCESSING_SSAO_H
