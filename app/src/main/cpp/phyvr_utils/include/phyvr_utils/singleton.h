//
// Created by samuel on 01/10/2025.
//

#ifndef PHYVR_SINGLETON_H
#define PHYVR_SINGLETON_H

template<typename T, typename... Args>
class Singleton {
public:
  static std::shared_ptr<T> get_singleton(Args... args) {
    if (!Singleton<T, Args...>::instance)
      Singleton<T, Args...>::instance = std::make_shared<T>(args...);
    return Singleton<T, Args...>::instance;
  }

  static void reset_singleton() {
    if (Singleton<T, Args...>::instance) Singleton<T, Args...>::instance = nullptr;
  }

  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton &) = delete;

private:
  static std::shared_ptr<T> instance;

  Singleton() = default;
  ~Singleton() = default;
};

template<typename T, typename... Args>
std::shared_ptr<T> Singleton<T, Args...>::instance = nullptr;

#endif// PHYVR_SINGLETON_H
