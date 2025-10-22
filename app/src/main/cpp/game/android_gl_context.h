//
// Created by samuel on 28/09/2025.
//

#ifndef ARENAI_ANDROID_GL_CONTEXT_H
#define ARENAI_ANDROID_GL_CONTEXT_H

#include <memory>

#include <android/native_activity.h>
#include <android/native_window.h>
#include <android_native_app_glue.h>
#include <EGL/egl.h>

#include <arenai_view/drawable.h>
#include <arenai_view/renderer.h>

class AndroidGLContext : public AbstractGLContext {
public:
    explicit AndroidGLContext(ANativeWindow *window, EGLDisplay display);
    EGLDisplay get_display() override;
    EGLSurface get_surface() override;
    EGLContext get_context() override;

private:
    EGLDisplay display;
    EGLSurface surface;
    EGLContext context;
};

#endif// ARENAI_ANDROID_GL_CONTEXT_H
