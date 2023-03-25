//
// Created by samuel on 19/03/2023.
//

#ifndef PHYVR_LOGGING_H
#define PHYVR_LOGGING_H

#include <android/log.h>

#define LOG_DEBUG(...)                                                         \
  ((void)__android_log_print(ANDROID_LOG_DEBUG, "phyvr", __VA_ARGS__))
#define LOG_INFO(...)                                                          \
  ((void)__android_log_print(ANDROID_LOG_INFO, "phyvr", __VA_ARGS__))
#define LOG_WARN(...)                                                          \
  ((void)__android_log_print(ANDROID_LOG_WARN, "phyvr", __VA_ARGS__))
#define LOG_ERROR(...)                                                         \
  ((void)__android_log_print(ANDROID_LOG_ERROR, "phyvr", __VA_ARGS__))

#endif // PHYVR_LOGGING_H
