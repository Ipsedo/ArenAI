//
// Created by samuel on 15/07/2026.
//

#ifndef ARENAI_POST_PROCESSING_BLOOM_BRIGHT_H
#define ARENAI_POST_PROCESSING_BLOOM_BRIGHT_H

#include "./effect.h"

namespace arenai::view {

    // half-res thresholded bright pass on the scene, publishes "bright"
    class BloomBrightEffect : public AbstractPostProcessingEffect {
    public:
        BloomBrightEffect(int width, int height);

        void render(FrameContext &context) override;
    };

}// namespace arenai::view

#endif// ARENAI_POST_PROCESSING_BLOOM_BRIGHT_H
