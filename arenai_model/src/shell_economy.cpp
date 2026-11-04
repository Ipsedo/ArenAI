//
// Created by samuel on 03/11/2026.
//

#include <cmath>

#include <arenai_model/shell_economy.h>

namespace arenai::model {

    namespace {
        int lerp_shells(const float progress, const int initial, const int final) {
            return static_cast<int>(std::lround(
                std::lerp(static_cast<float>(initial), static_cast<float>(final), progress)));
        };

    }// namespace

    ShellEconomy ShellEconomy::interpolate(const float progress, const ShellEconomy &final) const {
        return {
            .initial_shells = lerp_shells(progress, initial_shells, final.initial_shells),
            .shells_per_hit = lerp_shells(progress, shells_per_hit, final.shells_per_hit),
            .max_shells = lerp_shells(progress, max_shells, final.max_shells)};
    };
}// namespace arenai::model
