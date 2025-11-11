//
// Created by samuel on 01/10/2025.
//

#ifndef ARENAI_SINGLETON_H
#define ARENAI_SINGLETON_H

template<typename T, typename... Args>
class Singleton {
public:
    static std::shared_ptr<T> get_singleton(Args... args) {
        if (!instance) instance = std::make_shared<T>(args...);
        return instance;
    }

    static void reset_singleton() {
        if (instance) instance = nullptr;
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

#endif// ARENAI_SINGLETON_H
