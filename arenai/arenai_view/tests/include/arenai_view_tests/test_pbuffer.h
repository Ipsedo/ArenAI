//
// Created by samuel on 26/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_TEST_PBUFFER_H
#define ARENAI_TRAIN_HOST_TEST_PBUFFER_H

#include <cstdlib>

#include <gtest/gtest.h>

// The golden images record what one exact renderer rasterizes (llvmpipe on
// Ubuntu, see scripts/goldens_docker.sh). Comparing pixels anywhere else
// measures the local Mesa version, not a regression, so skip instead of
// failing. ARENAI_PINNED_RENDER_ENV is set by scripts/ci/render_env.sh.
#define ARENAI_SKIP_UNLESS_PINNED_RENDER_ENV()                                                     \
    do {                                                                                           \
        if (std::getenv("ARENAI_PINNED_RENDER_ENV") == nullptr)                                    \
            GTEST_SKIP() << "golden comparison needs the pinned render environment: "              \
                            "run ./scripts/goldens_docker.sh";                                     \
    } while (false)

struct image_size {
    int width;
    int height;
};

class PBufferParam : public testing::TestWithParam<image_size> {};

class PBufferDiffuseParam : public testing::TestWithParam<image_size> {};

class PBufferClearColorParam : public testing::TestWithParam<image_size> {};

class PBufferMultiFrameParam : public testing::TestWithParam<image_size> {};

#endif//ARENAI_TRAIN_HOST_TEST_PBUFFER_H
