//
// Created by samuel on 19/03/2023.
//

#ifndef ARENAI_LOGGING_H
#define ARENAI_LOGGING_H

#if ANDROID

#include <android/log.h>

#define LOG_DEBUG(...) ((void) __android_log_print(ANDROID_LOG_DEBUG, "arenai", __VA_ARGS__))
#define LOG_INFO(...) ((void) __android_log_print(ANDROID_LOG_INFO, "arenai", __VA_ARGS__))
#define LOG_WARN(...) ((void) __android_log_print(ANDROID_LOG_WARN, "arenai", __VA_ARGS__))
#define LOG_ERROR(...) ((void) __android_log_print(ANDROID_LOG_ERROR, "arenai", __VA_ARGS__))

#else

#include <iostream>

#define LOG_DEBUG(msg, ...)                                                                        \
    do { printf("[DEBUG] " msg "\n", ##__VA_ARGS__); } while (0)

#define LOG_INFO(msg, ...)                                                                         \
    do { printf("[INFO] " msg "\n", ##__VA_ARGS__); } while (0)

#define LOG_WARN(msg, ...)                                                                         \
    do { printf("[WARN] " msg "\n", ##__VA_ARGS__); } while (0)

#define LOG_ERROR(msg, ...)                                                                        \
    do { printf("[ERROR] " msg "\n", ##__VA_ARGS__); } while (0)

#endif

#endif// ARENAI_LOGGING_H
