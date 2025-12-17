//
// Created by samuel on 25/11/2025.
//

#ifndef ARENAI_TRAIN_HOST_TIMER_H
#define ARENAI_TRAIN_HOST_TIMER_H
#include <chrono>
#include <iostream>
#include <string>

class Timer {
public:
    explicit Timer(const char *name)
        : name(name), start(std::chrono::high_resolution_clock::now()) {}

    double get_duration() const {
        const auto end = std::chrono::high_resolution_clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        return static_cast<double>(us) / 1000.0;
    }

    friend std::ostream &operator<<(std::ostream &os, const Timer &timer) {
        os << timer.name << " : " << timer.get_duration() << " ms";
        return os;
    }

private:
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
};

#endif//ARENAI_TRAIN_HOST_TIMER_H
