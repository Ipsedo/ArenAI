//
// Created by samuel on 02/10/2025.
//

#ifndef PHYVR_AGENT_H
#define PHYVR_AGENT_H

#include <torch/script.h>

struct AgentState {
  torch::Tensor vision;
  torch::Tensor sensors;
};

class AbstractAgent : public torch::jit::Module {
public:
  virtual torch::Tensor act(AgentState state) = 0;
};

#endif // PHYVR_AGENT_H
