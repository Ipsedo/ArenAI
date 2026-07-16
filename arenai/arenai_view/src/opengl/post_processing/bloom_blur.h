//
// Created by samuel on 15/07/2026.
//

#ifndef ARENAI_POST_PROCESSING_BLOOM_BLUR_H
#define ARENAI_POST_PROCESSING_BLOOM_BLUR_H

#include "./effect.h"

namespace arenai::view {

    // separable gaussian blur of "bright" into a quarter-res ping-pong,
    // publishes "bloom"
    class BloomBlurEffect : public AbstractPostProcessingEffect {
    public:
        BloomBlurEffect(int width, int height);

        void render(FrameContext &context) override;
    };

}// namespace arenai::view

#endif// ARENAI_POST_PROCESSING_BLOOM_BLUR_H
