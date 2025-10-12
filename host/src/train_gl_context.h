//
// Created by samuel on 12/10/2025.
//

#ifndef PHYVR_TRAIN_HOST_TRAIN_GL_CONTEXT_H
#define PHYVR_TRAIN_HOST_TRAIN_GL_CONTEXT_H

#include <phyvr_view/renderer.h>

class TrainGlContext final : public AbstractGLContext {
public:
    TrainGlContext();

    EGLDisplay get_display() override;

    EGLSurface get_surface() override;

    EGLContext get_context() override;

private:
    EGLDisplay display;
    EGLContext context;
};

#endif//PHYVR_TRAIN_HOST_TRAIN_GL_CONTEXT_H
