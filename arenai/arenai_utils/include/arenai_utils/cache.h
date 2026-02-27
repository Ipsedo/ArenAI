//
// Created by samuel on 01/10/2025.
//
#ifndef ARENAI_CACHE_H
#define ARENAI_CACHE_H

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

template<class T>
class Cache {
public:
    bool exists(const std::string &name) const {
        std::scoped_lock lock(mutex);
        return cache.contains(name);
    }

    void add(const std::string &name, const T &obj) {
        std::scoped_lock lock(mutex);
        cache[name] = obj;
    }

    T get(const std::string &name) const {
        std::scoped_lock lock(mutex);
        auto it = cache.find(name);
        if (it == cache.end()) { throw std::out_of_range("Cache::get missing key: " + name); }
        return it->second;
    }

    void apply_on_items(const std::function<void(T)> &apply_fn) const {
        std::scoped_lock lock(mutex);
        for (const auto &[_, t]: cache) apply_fn(t);
    }

    void clear() {
        std::scoped_lock lock(mutex);
        cache.clear();
    }

private:
    mutable std::mutex mutex;
    std::map<std::string, T> cache;
};

#endif
