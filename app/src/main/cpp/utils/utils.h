//
// Created by samuel on 18/10/2025.
//

#ifndef PHYVR_UTILS_H
#define PHYVR_UTILS_H

#include <string>

#include <android_native_app_glue.h>
#include <jni.h>

void hide_system_status_bar(struct android_app *app);

JNIEnv *get_env(struct android_app *app);
jobject get_intent(JNIEnv *env, jobject activity);
int get_int_extra(JNIEnv *env, jobject intent, const char *key, int default_value = 0);

#endif//PHYVR_UTILS_H
