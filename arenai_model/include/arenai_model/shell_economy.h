#ifndef ARENAI_SHELL_ECONOMY_H
#define ARENAI_SHELL_ECONOMY_H

namespace arenai::model {
    struct ShellEconomy {
        int initial_shells;
        int shells_per_hit;
        int max_shells;
    };

    constexpr ShellEconomy GAME_SHELL_ECONOMY{
        .initial_shells = 10,
        .shells_per_hit = 2,
        .max_shells = 20};

}// namespace arenai::model

#endif//ARENAI_SHELL_ECONOMY_H
