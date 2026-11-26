---
name: training-loop-profile-2026-08
description: "Profil chrono de la boucle d'entraînement (2026-08-26, --cuda) : PPO = 55 % env.step (79 % attente rendu, iGPU Renoir par défaut) + 27 % update ; SAC (historique) = 61 % update"
metadata:
  node_type: memory
  type: project
  originSessionId: e3d7d3d0-7093-4288-a1b0-cdfc5cb26fe4
  modified: 2026-08-26T19:42:40.018Z
---

Benchmarks du 2026-08-26 (instrumentation chrono temporaire, retirée ensuite), config par
défaut (32 tanks, vision 256×128), `--cuda` (RTX 3060 Laptop). Rendu Vulkan choisi par
défaut = **iGPU AMD Renoir (RADV)**, pas la RTX.

## PPO (algo courant — voir [[ppo-reward-redesign-2026-08]]) — ~46 ms/step amorti

- `env->step()` : **55 %** (25,5 ms/step) — dont attente des threads de rendu offscreen
  **78 %** (19,8 ms, soit ~43 % du wall total), physique Jolt 13 % (3,4 ms), build_state 8 %.
- `trainer->step()` PPO : **27 %** — bursts de ~11 s tous les 900 steps (rollout_size).
  Par burst : actor 39 % (80 ms/minibatch de 1024), critic 28 % (57 ms/mb), GAE 14 %
  (1,5 s, forward sur tout le rollout), `select→to(device)` 12 % (24,6 ms/mb ≈ 100 Mo de
  visions uint8 pageables CPU→GPU ≈ 4 Go/s), get_rollout 8 %.
- reset épisode 7 % (~3 s/reset), act (inférence) 5,5 % (2,5 ms/step), conversions ~5 %.

## SAC (historique, abandonné) — ~83 ms/step amorti

- update SAC **61 %** (bursts de ~15,6 s / 256 steps, 128 epochs × ~122 ms : actor 40 %,
  critics 31 %, target 15 %, sample 13 %), env.step 29 %, reset 4 %, act 3 %.

## Expérience ARENAI_VK_DEVICE=NVIDIA (2026-08-26, mesurée)

Le backend offscreen préfère **délibérément l'iGPU** (prefer_integrated). Avec
`ARENAI_VK_DEVICE=NVIDIA` : render_wait **19,8 → 2,8 ms/step (×7)**, env.step
25,5 → ~7,3 ms/step ; profil env.step devient physics 41 %, render 38 %, build_state 17 %.
Stable avec un burst PPO sur la même RTX (boucle séquentielle : rendu et update ne se
recouvrent jamais → pas de contention ; seule la VRAM 6 Go est partagée).
`--vision_num_threads` 8/16/32 : aucun effet (GPU-bound, pas thread-bound). Gain global
estimé ≈ −40 % du temps d'entraînement. Non adopté par défaut — décision de Samuel requise.

## Mutex / multi-queue (2026-08-26, mesuré)

Un seul verrou Vulkan existe : `queue_mutex_` autour de `vkQueueSubmit` (une VkQueue,
`queueCount=1` dans device.cpp) — **non supprimable** (spec : queues externally
synchronized) ; le reste est déjà lock-free (DoubleBuffer atomique, command pool par
renderer, enregistrement 100 % parallèle). Coût mesuré en régime permanent :
**iGPU RADV = 511 µs/submit → ~16 ms/frame sérialisés ≈ tout le render_wait** (et RADV
n'expose qu'1 queue graphics → multi-queue impossible, l'iGPU est incurable) ;
**RTX = 19 µs/submit → ~0,6 ms/frame** (16 queues dispo, mais gain multi-queue marginal
en steady state ; aiderait surtout les resets, où le burst d'uploads fait ~45 s d'attente
mutex cumulée). Conclusion : le vrai fix des mutex, c'est ARENAI_VK_DEVICE=NVIDIA.

**Why:** savoir où optimiser : en PPO le goulot n°1 était le **rendu Vulkan sur l'iGPU**
(l'attente rendu seule ≈ 43 % du temps total), l'update PPO vient ensuite ; physique et
conversions marginales.

**How to apply:** ne pas re-mesurer de zéro — repartir de ces chiffres. Accélération
immédiate sans code : lancer le train avec `ARENAI_VK_DEVICE=NVIDIA`. Pistes plus lourdes
non validées par Samuel : batcher les 32 vues en une soumission (l'overhead par vue est
submit/sync, pas les pixels), pipeliner vision t−1, pin memory minibatches PPO. Toute
optimisation code = design à valider d'abord.
