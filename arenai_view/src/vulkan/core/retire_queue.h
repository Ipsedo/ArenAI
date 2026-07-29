//
// Created by samuel on 20/07/2026.
//

#ifndef ARENAI_VK_RETIRE_QUEUE_H
#define ARENAI_VK_RETIRE_QUEUE_H

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace arenai::view {

    // Deferred destruction of GPU-referenced objects: an item handed to
    // retire() may still be referenced by an in-flight frame, so it is kept
    // alive for lifetime_frames more tick() calls (one tick per rendered
    // frame) before being destroyed.
    template<typename T>
    class RetireQueue {
    public:
        explicit RetireQueue(const uint64_t lifetime_frames) : lifetime_frames_(lifetime_frames) {}

        // tags the item with the current frame; destroyed lifetime frames later
        void retire(std::unique_ptr<T> item) { retired_.emplace_back(frame_, std::move(item)); }

        // advances the frame counter and destroys every expired item
        void tick() {
            frame_++;
            std::erase_if(retired_, [this](const auto &entry) {
                return frame_ - entry.first >= lifetime_frames_;
            });
        }

        // immediate destruction of everything, once the GPU is known idle
        void drain_all() { retired_.clear(); }

    private:
        uint64_t lifetime_frames_;
        uint64_t frame_ = 0;
        std::vector<std::pair<uint64_t, std::unique_ptr<T>>> retired_;
    };

}// namespace arenai::view

#endif// ARENAI_VK_RETIRE_QUEUE_H
