# ArenAI training

Full C++ program to train ArenAI agent with LibTorch.

## Usage

Training is runnable through CLI with almost agents hyperparameters configurable.

See usage with :
```bash
cd /path/to/ArenAI
./build/arenai_agent/arenai_agent_train --help
```

Example : train liquid networks PPO with CUDA :
```bash
cd /path/to/ArenAI
./build/arenai_agent/arenai_agent_train --cuda --output_folder ./outputs/my_training --resources_folder ./resources ppo_liquid
```
