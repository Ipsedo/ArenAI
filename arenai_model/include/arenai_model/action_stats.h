//
// Created by samuel on 20/10/2025.
//

#ifndef ARENAI_ACTION_STATS_H
#define ARENAI_ACTION_STATS_H

#include <arenai_controller/inputs.h>

namespace arenai::model {

    class ActionStats {
    public:
        ActionStats();

        bool has_fire() const;
        float energy_consumed() const;
        void process_input(const controller::user_input &action);

    private:
        bool has_fire_;
        float energy_consumed_;
    };

}// namespace arenai::model

#endif//ARENAI_ACTION_STATS_H
