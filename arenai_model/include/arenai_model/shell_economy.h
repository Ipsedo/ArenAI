//
// Created by samuel on 03/11/2026.
//

#ifndef ARENAI_SHELL_ECONOMY_H
#define ARENAI_SHELL_ECONOMY_H

namespace arenai::model {
    struct ShellEconomy {
        int initial_shells;
        int shells_per_hit;
        int max_shells;

        ShellEconomy interpolate(float progress, const ShellEconomy &final) const;
    };

    constexpr ShellEconomy GAME_SHELL_ECONOMY{
        .initial_shells = 8,
        .shells_per_hit = 5,
        .max_shells = 16};

}// namespace arenai::model

#endif//ARENAI_SHELL_ECONOMY_H
