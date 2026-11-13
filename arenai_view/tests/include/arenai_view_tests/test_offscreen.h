#ifndef ARENAI_AGENT_HOST_TEST_OFFSCREEN_H
#define ARENAI_AGENT_HOST_TEST_OFFSCREEN_H

#include <cstdlib>

#include <gtest/gtest.h>

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

class OffscreenParam : public testing::TestWithParam<image_size> {};

class OffscreenDiffuseParam : public testing::TestWithParam<image_size> {};

class OffscreenClearColorParam : public testing::TestWithParam<image_size> {};

class OffscreenMultiFrameParam : public testing::TestWithParam<image_size> {};

#endif//ARENAI_AGENT_HOST_TEST_OFFSCREEN_H
