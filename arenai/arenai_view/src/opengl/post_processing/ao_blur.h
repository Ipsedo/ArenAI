//
// Created by samuel on 15/07/2026.
//

#ifndef ARENAI_POST_PROCESSING_AO_BLUR_H
#define ARENAI_POST_PROCESSING_AO_BLUR_H

#include "./effect.h"

namespace arenai::view {

    // blurs "ao_raw" to hide the SSAO spiral noise, publishes "ao"
    class AoBlurEffect : public AbstractPostProcessingEffect {
    public:
        AoBlurEffect(int width, int height);

        void render(FrameContext &context) override;
    };

}// namespace arenai::view

#endif// ARENAI_POST_PROCESSING_AO_BLUR_H
