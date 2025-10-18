//
// Created by samuel on 18/10/2025.
//

#include "./utils.h"

#include <android/native_activity.h>

void hide_system_status_bar(struct android_app *app) {
    ANativeActivity *activity = app->activity;
    JavaVM *vm = activity->vm;
    JNIEnv *env = nullptr;

    vm->AttachCurrentThread(&env, nullptr);

    jobject activityObj = activity->clazz;

    jclass activityClass = env->GetObjectClass(activityObj);
    jmethodID getWindow = env->GetMethodID(activityClass, "getWindow", "()Landroid/view/Window;");

    jobject window = env->CallObjectMethod(activityObj, getWindow);
    jclass windowClass = env->GetObjectClass(window);

    jmethodID getDecorView = env->GetMethodID(windowClass, "getDecorView", "()Landroid/view/View;");
    jobject decorView = env->CallObjectMethod(window, getDecorView);

    jclass viewClass = env->GetObjectClass(decorView);

    jmethodID setSystemUiVisibility = env->GetMethodID(viewClass, "setSystemUiVisibility", "(I)V");

    const int SYSTEM_UI_FLAG_FULLSCREEN = 0x00000004;
    const int SYSTEM_UI_FLAG_HIDE_NAVIGATION = 0x00000002;
    const int SYSTEM_UI_FLAG_IMMERSIVE_STICKY = 0x00001000;

    int flags = SYSTEM_UI_FLAG_FULLSCREEN | SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | SYSTEM_UI_FLAG_IMMERSIVE_STICKY;

    // Applique le mode immersif
    env->CallVoidMethod(decorView, setSystemUiVisibility, flags);

    vm->DetachCurrentThread();
}
