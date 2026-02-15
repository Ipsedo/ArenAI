//
// Created by samuel on 15/02/2026.
//

#ifndef ARENAI_TRAIN_HOST_DOUBLE_BUFFER_H
#define ARENAI_TRAIN_HOST_DOUBLE_BUFFER_H

#include <array>
#include <atomic>
#include <cstddef>
#include <utility>

template<class T>
class DoubleBuffer {
public:
    explicit DoubleBuffer(const T &initial) : buf_{initial, initial}, front_{0} {}

    DoubleBuffer(DoubleBuffer &&other) noexcept { move_from(other); }

    DoubleBuffer &operator=(DoubleBuffer &&other) noexcept {
        if (this != &other) move_from(other);
        return *this;
    }

    DoubleBuffer(const DoubleBuffer &) = delete;
    DoubleBuffer &operator=(const DoubleBuffer &) = delete;

    void write(const T &to_write) {
        const std::size_t f = front_.load(std::memory_order_relaxed);
        const std::size_t b = 1u - f;

        buf_[b] = to_write;
        front_.store(b, std::memory_order_release);
    }

    T get() const {
        const std::size_t f = front_.load(std::memory_order_acquire);
        return buf_[f];
    }

private:
    std::array<T, 2> buf_;
    std::atomic<std::size_t> front_;

    void move_from(DoubleBuffer &other) noexcept {
        const std::size_t f = other.front_.load(std::memory_order_relaxed);
        buf_[0] = std::move(other.buf_[0]);
        buf_[1] = std::move(other.buf_[1]);
        front_.store(f, std::memory_order_relaxed);
    }
};

#endif//ARENAI_TRAIN_HOST_DOUBLE_BUFFER_H
