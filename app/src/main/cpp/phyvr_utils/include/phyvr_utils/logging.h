//
// Created by samuel on 19/03/2023.
//

#ifndef PHYVR_LOGGING_H
#define PHYVR_LOGGING_H

#if ANDROID

#include <android/log.h>

#define LOG_DEBUG(...) ((void) __android_log_print(ANDROID_LOG_DEBUG, "phyvr", __VA_ARGS__))
#define LOG_INFO(...) ((void) __android_log_print(ANDROID_LOG_INFO, "phyvr", __VA_ARGS__))
#define LOG_WARN(...) ((void) __android_log_print(ANDROID_LOG_WARN, "phyvr", __VA_ARGS__))
#define LOG_ERROR(...) ((void) __android_log_print(ANDROID_LOG_ERROR, "phyvr", __VA_ARGS__))

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

#endif// PHYVR_LOGGING_H
