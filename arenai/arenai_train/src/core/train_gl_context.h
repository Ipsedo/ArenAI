//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_TRAIN_GL_CONTEXT_H
#define ARENAI_TRAIN_HOST_TRAIN_GL_CONTEXT_H

#include <arenai_view/renderer.h>

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

#endif//ARENAI_TRAIN_HOST_TRAIN_GL_CONTEXT_H
