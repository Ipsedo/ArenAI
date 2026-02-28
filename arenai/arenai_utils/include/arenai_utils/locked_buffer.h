//
// Created by samuel on 27/02/2026.
//

#ifndef ARENAI_TRAIN_HOST_ATOMIC_SNAPSHOT_H
#define ARENAI_TRAIN_HOST_ATOMIC_SNAPSHOT_H

#include <atomic>
#include <memory>

template<class T>
class LockedBuffer {
public:
    explicit LockedBuffer(T initial) : value_(std::move(initial)) {}

    void write(T v) {
        std::lock_guard lock(m_);
        value_ = std::move(v);
    }

    T read_copy() const {
        std::lock_guard lock(m_);
        return value_;
    }

    template<class F>
    auto with_read(F &&f) const -> decltype(f(std::declval<const T &>())) {
        std::lock_guard lock(m_);
        return std::forward<F>(f)(value_);
    }

private:
    mutable std::mutex m_;
    T value_;
};

#endif//ARENAI_TRAIN_HOST_ATOMIC_SNAPSHOT_H
