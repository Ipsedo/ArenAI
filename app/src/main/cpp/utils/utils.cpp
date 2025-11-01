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

/*
 * Intent
 */

JNIEnv *get_env(struct android_app *app) {
    JNIEnv *env = nullptr;
    app->activity->vm->AttachCurrentThread(&env, nullptr);
    return env;
}

jobject get_intent(JNIEnv *env, jobject activity) {
    jclass cls = env->GetObjectClass(activity);
    jmethodID mid = env->GetMethodID(cls, "getIntent", "()Landroid/content/Intent;");
    jobject intent = env->CallObjectMethod(activity, mid);
    env->DeleteLocalRef(cls);
    return intent;
}

int get_int_extra(JNIEnv *env, jobject intent, const char *key, int default_value) {
    jclass cls = env->GetObjectClass(intent);
    jmethodID mid = env->GetMethodID(cls, "getIntExtra", "(Ljava/lang/String;I)I");
    jstring jkey = env->NewStringUTF(key);
    jint res = env->CallIntMethod(intent, mid, jkey, (jint) default_value);
    env->DeleteLocalRef(jkey);
    env->DeleteLocalRef(cls);
    return (int) res;
}

std::string
get_string_extra(JNIEnv *env, jobject intent, const char *key, const std::string &default_value) {
    jclass cls = env->GetObjectClass(intent);
    jmethodID mid =
        env->GetMethodID(cls, "getStringExtra", "(Ljava/lang/String;)Ljava/lang/String;");
    jstring jkey = env->NewStringUTF(key);

    jstring jres = (jstring) env->CallObjectMethod(intent, mid, jkey);
    std::string result = default_value;

    const char *cstr = env->GetStringUTFChars(jres, nullptr);
    result = cstr;
    env->ReleaseStringUTFChars(jres, cstr);
    env->DeleteLocalRef(jres);

    env->DeleteLocalRef(jkey);
    env->DeleteLocalRef(cls);
    return result;
}
