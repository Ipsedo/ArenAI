//
// Created by samuel on 01/10/2025.
//

#ifndef ARENAI_CACHE_H
#define ARENAI_CACHE_H

#include <functional>
#include <map>
#include <memory>

template<class T>
class Cache {
private:
    std::map<std::string, T> cache;

public:
    Cache() : cache() {}

    bool exists(const std::string &name) { return cache.contains(name); }

    void add(const std::string &name, const T &obj) { cache[name] = obj; }
    T get(const std::string &name) { return cache[name]; }

    void apply_on_items(const std::function<void(T)> apply_fn) {
        for (auto [_, t]: cache) apply_fn(t);
    }

    void clear() { cache.clear(); }
};

#endif// ARENAI_CACHE_H
