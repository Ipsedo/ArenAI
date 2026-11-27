# Usage

Build first: [BUILD.md](BUILD.md). Code layout: [ARCHITECTURE.md](ARCHITECTURE.md).

All commands run from the repo root, on a `./build_linux.sh` tree (`build/`).

## Play

```bash
./build/arenai_desktop/arenai_desktop
```

```bash
./build/arenai_desktop/arenai_desktop --window_width 2560 --window_height 1440
```

| option | default |
|---|---|
| `--window_width` | `1920` |
| `--window_height` | `1080` |

Resources are read from `resources/` next to the executable (copied there by the build).

Bundled agents (picked in the menu):

```
resources/trained_models/<ppo|ppo_liquid|ppg>/config.json
resources/trained_models/<ppo|ppo_liquid|ppg>/save_K/actor.pt
```

Your own agent: in the menu's AI page, pick the agent type (`ppo`, `ppo_liquid`, `ppg`) and a
`save_K/` folder holding `actor.pt`; `config.json` is taken from that folder or its parent.

## Train

Every training runs 32 tanks driven by the same agent in one arena, each seeing only its own
camera; the spawn area grows from 250 m to 1000 m as they learn to hit each other.

`ppo` — Proximal Policy Optimization, feed-forward actor and critic (CNN + MLP), default options:

```bash
./build/arenai_agent/arenai_agent_train \
    --output_folder outputs/train_001_ppo \
    --resources_folder resources \
    --cuda \
    ppo
```

`ppo_liquid` — PPO with a recurrent liquid cell of 128 neurons as memory, trained by truncated
backpropagation through time on chunks of 30 steps:

```bash
./build/arenai_agent/arenai_agent_train \
    --output_folder outputs/train_001_ppo_liquid \
    --resources_folder resources \
    --cuda \
    ppo_liquid --neuron_number 128 --chunk_size 30
```

`ppg` — Phasic Policy Gradient: 8 PPO policy phases, then one auxiliary phase distilling the
value into the actor's encoder:

```bash
./build/arenai_agent/arenai_agent_train \
    --output_folder outputs/train_001_ppg \
    --resources_folder resources \
    --cuda \
    ppg --n_pi 8 --beta_clone 1.0
```

`sac` — Soft Actor-Critic, off-policy: a replay buffer of 200 000 transitions, 2 gradient steps
per environment step:

```bash
./build/arenai_agent/arenai_agent_train \
    --output_folder outputs/train_001_sac \
    --resources_folder resources \
    --cuda \
    sac --buffer_size 200000 --gradient_steps 2
```

Common options go **before** the agent name, agent options **after**. No agent name = `ppo`.

```bash
./build/arenai_agent/arenai_agent_train --help
./build/arenai_agent/arenai_agent_train ppo_liquid --help
```

### Output

```
outputs/train_001_ppo/
├── config.json      # train + environment + agent options of the run
├── metrics.csv      # ';' separated
└── save_K/          # actor.pt, critic.pt, optimizers, …
```

The folder is exactly `--output_folder`; `train_NNN_<agent>` is only a naming convention.

### Common options

| option | default | meaning |
|---|---|---|
| `--output_folder` | required | run folder |
| `--resources_folder` | required | game assets folder |
| `--cuda` | off | networks on GPU |
| `--nb_episodes` | `30000` | episodes to play |
| `--max_episode_steps` | `5400` | step cap per episode |
| `--save_every` | `27000` | steps between two `save_K/` |
| `--nb_tanks` | `32` | tanks in the arena |
| `--wanted_frequency` | `0.0333` | physics step, seconds |
| `--action_repeat` | `3` | frames per agent decision |
| `--vision_num_threads` | CPU threads | vision rendering threads |
| `--shells` | `10` | initial shells per tank |
| `--shells_per_hit` | `2` | shells gained per hit |
| `--max_shells` | `30` | shell reserve cap |
| `--initial_spawn_width` / `_height` | `250` | spawn area at start, metres |
| `--final_spawn_width` / `_height` | `1000` | spawn area at curriculum end |
| `--curriculum_delta` | `50` | spawn side increment, metres |
| `--curriculum_ratio_low` | `0.3` | curriculum lower threshold |
| `--curriculum_ratio_high` | `0.5` | curriculum upper threshold |
| `--curriculum_probe_window` | `8` | episodes per curriculum decision |
| `--curriculum_boundary_proba` | `0.5` | probability to play the bound |

### Agent options — `ppo`, `ppo_liquid`, `ppg`

| option | default | meaning |
|---|---|---|
| `--actor_learning_rate` | `1e-4` | actor learning rate |
| `--critic_learning_rate` | `3e-4` | critic learning rate |
| `--gamma` | `0.998` | discount factor |
| `--gae_lambda` | `0.95` | GAE λ |
| `--clip_epsilon` | `0.2` | PPO clip range |
| `--target_kl` | `0.05` | KL early-stop threshold |
| `--grad_norm_max` | `0.5` | gradient norm clip |
| `--rollout_size` | `300` | steps per rollout |
| `--actor_epochs` | `4` | actor epochs per rollout |
| `--critic_epochs` | `8` | critic epochs per rollout |
| `--actor_batch_size` | `2400` | actor optimisation batch |
| `--critic_batch_size` | `512` | critic optimisation batch |
| `--actor_gpu_batch_size` | `512` (`ppg`: `256`) | actor forward chunk |
| `--critic_gpu_batch_size` | `512` (`ppg`: `256`) | critic forward chunk |
| `--initial_sigma` | `0.3` | initial action std |
| `--initial_fire_proba` | `0.5` | initial fire probability |
| `--initial_zoom_proba` | `0.5` | initial zoom probability |
| `--initial_alpha` | `0.001` | initial entropy coefficient |
| `--alpha_learning_rate` | `3e-4` | entropy coefficient learning rate |
| `--continuous_target_entropy` | `[-1,-1,-1,-1]` | target entropy per continuous action |
| `--discrete_target_proba` | `0.01` | discrete entropy target |
| `--hidden_size_sensors` | `64` | proprioception encoder width |
| `--actor_hidden_sizes` | `[256,128]` | actor MLP (`ppo`, `ppg`) |
| `--critic_hidden_sizes` | `[256,128]` | critic MLP (`ppo`, `ppg`) |
| `--vision_channels` | 7 conv layers | vision CNN `[in,out]` pairs |
| `--group_norm_nums` | `[2,4,8,16,16,16,16]` | groups per vision layer |
| `--panorama_channels` | `[[3,16],[16,32],[32,64]]` | panorama CNN `[in,out]` pairs |
| `--panorama_group_norm_nums` | `[2,4,8]` | groups per panorama layer |
| `--panorama_output_channels` | `4` | panorama channels kept per bearing column before its logit |
| `--panorama_mask_learning_rate` | `3e-4` | learning rate of the panorama mask loss |
| `--panorama_mask_batch_size` | `512` | panorama mask optimisation batch |
| `--panorama_mask_epochs` | `1` | passes over the new panoramas per update |
| `--metric_window_size` | `256` | metric averaging window |

`ppo_liquid` only:

| option | default | meaning |
|---|---|---|
| `--neuron_number` | `128` | liquid cell neurons |
| `--unfolding_steps` | `6` | ODE unfolding steps |
| `--delta_t` | `0.1` | liquid cell time step |
| `--chunk_size` | `30` | TBPTT chunk length |

`ppg` only:

| option | default | meaning |
|---|---|---|
| `--n_pi` | `8` | policy phases per auxiliary phase |
| `--actor_aux_epochs` | `4` | auxiliary actor epochs |
| `--critic_aux_epochs` | `4` | auxiliary critic epochs |
| `--beta_clone` | `1.0` | behaviour cloning weight |

### Agent options — `sac`

Shares the network, `--panorama_mask_*` and `--initial_*` options above.

| option | default | meaning |
|---|---|---|
| `--actor_learning_rate` | `3e-4` | actor learning rate |
| `--critic_learning_rate` | `3e-4` | critic learning rate |
| `--alpha_learning_rate` | `3e-5` | entropy coefficient learning rate |
| `--initial_alpha` | `5e-4` | initial entropy coefficient |
| `--gamma` | `0.99` | discount factor |
| `--n_step` | `10` | n-step return length |
| `--tau` | `0.005` | target network update rate |
| `--target_sigma` | `0.1` | continuous entropy target, as std |
| `--discrete_target_entropy_ratio` | `0.3` | discrete entropy target ratio |
| `--buffer_size` | `200000` | replay buffer capacity |
| `--armed_fraction` | `0.7` | batch share of armed transitions |
| `--learning_starts` | `5000` | transitions before first update |
| `--gradient_steps` | `2` | updates per environment step |
| `--batch_size` | `512` | optimisation batch |
| `--gpu_batch_size` | `1024` | forward chunk |

List options are JSON:

```bash
./build/arenai_agent/arenai_agent_train \
    --output_folder outputs/train_002_ppo \
    --resources_folder resources \
    ppo --actor_hidden_sizes '[512,256]' --vision_channels '[[3,16],[16,32],[32,64]]' \
        --group_norm_nums '[2,4,8]'
```

Vision size (256×128) and panorama size (512×32) are compile-time constants
(`arenai_core/include/arenai_core/types.h`).

Resuming a training from a checkpoint is not supported by the CLI.

## Evaluate

All three read `<run_folder>/config.json` and `<run_folder>/<save>/`.

### Match — game rules

```bash
mkdir -p outputs/train_001_ppo_liquid/matches
./build/arenai_agent/arenai_agent_match ppo_liquid \
    --run_folder outputs/train_001_ppo_liquid \
    --save save_10 \
    --resources_folder resources \
    --output outputs/train_001_ppo_liquid/matches/save_10 \
    --cuda
```

Positional agent: `ppo`, `ppo_liquid` or `ppg`. Writes `<output>_tanks.csv`,
`<output>_player.csv`, `<output>_matches.csv`.

| option | default | meaning |
|---|---|---|
| `--nb_matches` | `8` | matches to play |
| `--nb_tanks` | `16` | enemy tanks |
| `--spawn_side` | `500` | spawn square side, metres |
| `--max_seconds` | `300` | match time limit |
| `--seed` | `0` | seed of the first match |
| `--vision_num_threads` | `8` | vision rendering threads |
| `--cuda` | off | agent on GPU |
| `--approach_oracle` | off | scripted chassis approach |
| `--approach_steer` | `-2` | oracle steering gain |
| `--approach_throttle` | `1` | oracle throttle |
| `--approach_stop_distance` | `40` | oracle stop distance, metres |
| `--acquire_oracle` | off | scripted off-field target acquisition |
| `--acquire_slew` | `-1` | oracle slew command |
| `--acquire_half_field` | `39.5` | oracle half field, degrees |

### Probe — training rules

```bash
mkdir -p outputs/train_001_ppo_liquid/probes
./build/arenai_agent/arenai_agent_probe \
    --run_folder outputs/train_001_ppo_liquid \
    --save save_10 \
    --resources_folder resources \
    --output outputs/train_001_ppo_liquid/probes/save_10.csv \
    --cuda
```

Agent family read from `config.json`: `ppo_liquid`, `ppg`, `sac`. Also writes
`save_10_shots.csv` and `save_10_rewards.csv` next to `--output`.

| option | default | meaning |
|---|---|---|
| `--nb_episodes` | `4` | episodes to play |
| `--spawn_side` | `550` | spawn square side, metres |
| `--shells` / `--shells_per_hit` / `--max_shells` | `10` / `2` / `30` | shell economy |
| `--deterministic` | off | act at the distribution mode |
| `--zoom` | `policy` | `policy`, `on` or `off` |
| `--features <file>` | none | dump encoder features (binary) |
| `--frames <file>` | none | dump vision frames (binary) |
| `--frames_every` | `5` | frame dump period, steps |
| `--oracle` | off | scripted aim and fire |
| `--oracle_fov` | `360` | oracle field of view, degrees |
| `--oracle_tolerance` | `1` | oracle fire tolerance, degrees |
| `--oracle_policy_yaw` | off | yaw left to the policy |
| `--oracle_policy_pitch` | off | pitch left to the policy |
| `--oracle_policy_fire` | off | fire left to the policy |
| `--oracle_camera_center` | off | oracle aims with the camera centre |
| `--detrend_canon_x_steps` | `0` | canon yaw detrending window |
| `--vision_num_threads` | CPU threads | vision rendering threads |
| `--cuda` | off | agent on GPU |

### Sigma gradients — `ppo_liquid` only

```bash
./build/arenai_agent/arenai_agent_sigma_gradients \
    --run_folder outputs/train_001_ppo_liquid \
    --save save_10 \
    --resources_folder resources \
    --output outputs/train_001_ppo_liquid/probes/save_10_sigma_gradients.csv \
    --cuda
```

| option | default | meaning |
|---|---|---|
| `--nb_episodes` | `2` | episodes to play |
| `--spawn_side` | `1000` | spawn square side, metres |
| `--critic_inputs <file>` | none | dump critic inputs (binary) |
| `--frames <file>` | none | dump vision frames (binary) |
| `--frames_every` | `3` | frame dump period, steps |
| `--frames_fired` | off | frame dump restricted to shots |
| `--kl_breakdown <file>` | none | per-action KL CSV |
| `--kl_record_every` | `4` | KL record period |
| `--vision_num_threads` | CPU threads | vision rendering threads |
| `--cuda` | off | agent on GPU |

### Render bench

```bash
./build/arenai_agent/arenai_agent_render_bench \
    --resources_folder resources \
    --frames /tmp/reset_frames.bin
```

Prints `startup_ms`, `step_mean_ms`, `step_p50_ms`, `step_p90_ms`, `step_p99_ms`, `max_rss_mib`.

| option | default | meaning |
|---|---|---|
| `--nb_tanks` | `32` | tanks |
| `--steps` | `300` | timed steps |
| `--warmup` | `20` | untimed steps first |
| `--action_repeat` | `3` | frames per step |
| `--vision_num_threads` | CPU threads | vision rendering threads |
| `--seed` | `42` | environment and action seed |
| `--frames <file>` | none | dump the reset visions and panoramas (raw `uint8`, per tank) |

## Python tooling

```bash
python -m venv .venv
.venv/bin/pip install -r arenai_agent/python/requirements.txt
```

### Plot metrics

```bash
cd arenai_agent/python
../../.venv/bin/python -m arenai_visualize_metrics \
    ../../outputs/train_001_ppo/metrics.csv -o metrics.png
```

`-o` defaults to `arenai_agent_metrics.png`.

### Convert old checkpoints

Both need `torch` (not in `requirements.txt`): `.venv/bin/pip install torch`.

`mu` head `Sequential(Linear)` → `Linear`, written to a `converted/` sibling folder:

```bash
.venv/bin/python arenai_agent/python/convert_mu_linear.py \
    outputs/train_001_ppo_liquid/save_*/actor.pt
```

Overwrite, original kept as `actor.pt.bak`:

```bash
.venv/bin/python arenai_agent/python/convert_mu_linear.py --in-place \
    outputs/train_001_ppo_liquid/save_9/actor.pt
```

`LiquidRecurrent` / `LiquidCell` fusion — set `OLD_CHECKPOINT` in the first cell:

```bash
.venv/bin/pip install jupyter
.venv/bin/jupyter notebook arenai_agent/python/notebooks/convert_liquid_cell_fusion.ipynb
```

## Environment variables

| variable | applies to | value |
|---|---|---|
| `ARENAI_VK_DEVICE` | agent vision (offscreen): game, training, probes | device index, or substring of the device name |
| `ARENAI_VK_DEVICE_WINDOW` | game window | device index, or substring of the device name |
| `ARENAI_VK_VALIDATION` | every Vulkan instance | set to anything: validation layers on, if installed |

```bash
ARENAI_VK_DEVICE=NVIDIA ./build/arenai_agent/arenai_agent_train \
    --output_folder outputs/train_001_ppo --resources_folder resources --cuda ppo
```

```bash
ARENAI_VK_DEVICE_WINDOW=0 ARENAI_VK_DEVICE=1 ./build/arenai_desktop/arenai_desktop
```

An out-of-range index or an unmatched name aborts at startup.
