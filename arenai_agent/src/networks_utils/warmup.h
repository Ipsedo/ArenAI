#ifndef ARENAI_WARMUP_H
#define ARENAI_WARMUP_H

#include <torch/torch.h>

class CosineAnnealing {
public:
    CosineAnnealing(float initial_value, float final_value, int64_t warmup_env_step);

    float value() const;
    void step(int64_t nb_env_steps);

private:
    float initial;
    float final;
    int64_t warmup_env_step;

    long current_step;
};

#endif//ARENAI_WARMUP_H
