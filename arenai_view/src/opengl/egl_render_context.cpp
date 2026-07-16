//
// Created by samuel on 08/07/2026.
//

#include "./egl_render_context.h"

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <EGL/eglext.h>

#include <arenai_utils/logging.h>

namespace {

    bool has_extension(const char *extensions, const char *name) {
        return extensions != nullptr && std::strstr(extensions, name) != nullptr;
    }

    /*
     * The same physical NVIDIA card can be exposed as several EGL devices (proprietary
     * driver, Mesa/zink) without a common marker, so check the PCI vendor id (0x10de)
     * of the underlying DRM node instead of EGL device extensions.
     */
    bool is_nvidia_drm_device(const char *drm_file) {
        if (drm_file == nullptr) return false;
        const auto node = std::filesystem::path(drm_file).filename().string();
        std::ifstream vendor_file("/sys/class/drm/" + node + "/device/vendor");
        std::string vendor;
        if (!(vendor_file >> vendor)) return false;
        return vendor == "0x10de";
    }

    /*
     * Pick the integrated (non-NVIDIA) GPU explicitly, so headless contexts don't follow
     * the process-wide vendor selection (e.g. prime-run). Returns EGL_NO_DISPLAY when
     * EGL_EXT_platform_device is unavailable (e.g. Android) or no such GPU exists.
     */
    EGLDisplay get_integrated_gpu_display() {
        if (!has_extension(
                eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS), "EGL_EXT_platform_device"))
            return EGL_NO_DISPLAY;

        const auto query_devices =
            reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
        const auto query_device_string = reinterpret_cast<PFNEGLQUERYDEVICESTRINGEXTPROC>(
            eglGetProcAddress("eglQueryDeviceStringEXT"));
        const auto get_platform_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
            eglGetProcAddress("eglGetPlatformDisplayEXT"));
        if (query_devices == nullptr || query_device_string == nullptr
            || get_platform_display == nullptr)
            return EGL_NO_DISPLAY;

        EGLDeviceEXT devices[16];
        EGLint nb_devices = 0;
        if (query_devices(16, devices, &nb_devices) != EGL_TRUE) return EGL_NO_DISPLAY;

        for (EGLint i = 0; i < nb_devices; i++) {
            const char *device_extensions = query_device_string(devices[i], EGL_EXTENSIONS);
            if (has_extension(device_extensions, "EGL_NV_device_cuda")
                || has_extension(device_extensions, "EGL_MESA_device_software")
                || !has_extension(device_extensions, "EGL_EXT_device_drm"))
                continue;

            const char *drm_file = query_device_string(devices[i], EGL_DRM_DEVICE_FILE_EXT);
            if (is_nvidia_drm_device(drm_file)) continue;

            const auto display = get_platform_display(EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
            if (display == EGL_NO_DISPLAY) continue;

            LOG_INFO(
                "Headless EGL: using integrated GPU (%s)",
                drm_file != nullptr ? drm_file : "unknown DRM device");
            return display;
        }

        return EGL_NO_DISPLAY;
    }

}// namespace

namespace arenai::view {

    /*
     * EglRenderContext
     */

    void EglRenderContext::make_current() {
        if (eglMakeCurrent(get_display(), get_surface(), get_surface(), get_context()) != EGL_TRUE)
            throw std::runtime_error("Can't make context");

        // Desktop GL core profile has no default vertex array object (unlike
        // GLES3), so every glVertexAttribPointer call needs a VAO bound. One
        // VAO per context is enough: attribute state is reconfigured on each
        // draw, so this just restores the implicit-VAO behaviour of GLES.
        if (vao_ == 0) glGenVertexArrays(1, &vao_);
        glBindVertexArray(vao_);
    }

    void EglRenderContext::release_current() {
        eglMakeCurrent(get_display(), EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }

    /*
     * HeadlessEglContext
     */

    HeadlessEglContext::HeadlessEglContext() {
        display = get_integrated_gpu_display();

        EGLint major, minor;
        if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) {
            display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
            if (display == EGL_NO_DISPLAY) throw std::runtime_error("eglGetDisplay() failed");

            if (!eglInitialize(display, &major, &minor))
                throw std::runtime_error("eglInitialize() failed");
        }

        const EGLint cfg_attribs[] = {
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_BIT,
            EGL_SURFACE_TYPE,
            EGL_PBUFFER_BIT,
            EGL_RED_SIZE,
            8,
            EGL_GREEN_SIZE,
            8,
            EGL_BLUE_SIZE,
            8,
            EGL_ALPHA_SIZE,
            8,
            EGL_DEPTH_SIZE,
            16,
            EGL_STENCIL_SIZE,
            8,
            EGL_SAMPLES,
            0,
            EGL_NONE};

        EGLConfig configs[64];
        EGLint ncfg = 0;
        if (!eglChooseConfig(display, cfg_attribs, configs, 64, &ncfg) || ncfg < 1)
            throw std::runtime_error("eglChooseConfig() failed");

        EGLConfig config = nullptr;
        for (EGLint i = 0; i < ncfg; ++i) {
            EGLint r = 0, g = 0, b = 0, a = 0;
            eglGetConfigAttrib(display, configs[i], EGL_RED_SIZE, &r);
            eglGetConfigAttrib(display, configs[i], EGL_GREEN_SIZE, &g);
            eglGetConfigAttrib(display, configs[i], EGL_BLUE_SIZE, &b);
            eglGetConfigAttrib(display, configs[i], EGL_ALPHA_SIZE, &a);
            if (r == 8 && g == 8 && b == 8 && a == 8) {
                config = configs[i];
                break;
            }
        }
        if (config == nullptr) throw std::runtime_error("No 8-bit RGBA EGLConfig available");

        eglBindAPI(EGL_OPENGL_API);
        constexpr EGLint ctx_attribs[] = {
            EGL_CONTEXT_MAJOR_VERSION,
            3,
            EGL_CONTEXT_MINOR_VERSION,
            3,
            EGL_CONTEXT_OPENGL_PROFILE_MASK,
            EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            EGL_NONE};
        context = eglCreateContext(display, config, EGL_NO_CONTEXT, ctx_attribs);
        if (context == EGL_NO_CONTEXT) throw std::runtime_error("eglCreateContext() failed");
    }

    EGLDisplay HeadlessEglContext::get_display() { return display; }

    EGLSurface HeadlessEglContext::get_surface() { return EGL_NO_SURFACE; }

    EGLContext HeadlessEglContext::get_context() { return context; }

    /*
     * NativeEglContext
     */

    NativeEglContext::NativeEglContext(
        const EGLDisplay &display, const EGLSurface &surface, const EGLContext &context)
        : display(display), surface(surface), context(context) {}

    EGLDisplay NativeEglContext::get_display() { return display; }

    EGLSurface NativeEglContext::get_surface() { return surface; }

    EGLContext NativeEglContext::get_context() { return context; }

}// namespace arenai::view
