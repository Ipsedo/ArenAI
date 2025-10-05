//
// Created by samuel on 28/09/2025.
//

#ifndef PHYVR_ANDROID_GL_CONTEXT_H
#define PHYVR_ANDROID_GL_CONTEXT_H

#include <memory>

#include <android/native_activity.h>
#include <android/native_window.h>
#include <android_native_app_glue.h>
#include <EGL/egl.h>

#include <phyvr_view/drawable.h>
#include <phyvr_view/renderer.h>

class AndroidGLContext : public AbstractGLContext {
public:
    explicit AndroidGLContext(ANativeWindow *window);
    EGLDisplay get_display() override;
    EGLSurface get_surface() override;
    EGLContext get_context() override;

private:
    EGLDisplay display;
    EGLSurface surface;
    EGLContext context;
};

#endif// PHYVR_ANDROID_GL_CONTEXT_H
