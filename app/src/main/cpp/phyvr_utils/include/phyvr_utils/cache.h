//
// Created by samuel on 01/10/2025.
//

#ifndef PHYVR_CACHE_H
#define PHYVR_CACHE_H

#include <map>
#include <memory>

template <template <class> class C, class Type>
static std::shared_ptr<C<Type>> get_singleton() {
  if (C<Type>::singleton_cache == std::nullptr_t())
    C<Type>::singleton_cache = std::shared_ptr<C<Type>>(new C<Type>());
  return C<Type>::singleton_cache;
}

template <class T> class Cache {
private:
  std::map<std::string, T> cache;

public:
  Cache() : cache() {}

  bool exists(const std::string &name) {
    return cache.find(name) != cache.end();
  }

  void add(const std::string &name, const T &obj) { cache[name] = obj; }
  T get(const std::string &name) { return cache[name]; }
};

#endif // PHYVR_CACHE_H
