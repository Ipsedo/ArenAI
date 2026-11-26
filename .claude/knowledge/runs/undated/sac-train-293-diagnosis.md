---
name: sac-train-293-diagnosis
description: "Diagnostic run 293 SAC — morts par famine à 97 %, Q gonflé ×25 par la normalisation du reward"
metadata:
  node_type: memory
  type: project
  originSessionId: 3b30ce76-ccc8-4012-84b6-30ff71fc73a6
  modified: 2026-08-17T08:20:09.310Z
---

Analyse de `outputs/train_293_sac` (376 k steps env, ~455 épisodes, 2026-08-17) :

- `s_μ` plat à 27.5 s alors que `max_frames_without_hit` = 30 s → **~97 % des tanks meurent du
  timer de famine**, pas au combat (1–2 kills/épisode pour 32 tanks). Le signal dominant est un −1
  quasi constant et non prédictible.
- `hit/fire` = 0.42 % → 0.91 % sur la run : apprentissage réel mais très lent. 248 fenêtres sur 421
  contiennent zéro hit.
- `π_μ` ≈ −Q̄ (α ≈ 0.002 rend le terme d'entropie négligeable) et monte à 15.5 : Q est gonflé d'un
  facteur ~25 par `reward_scale()` (écart-type de vie d'un reward sparse ≈ 0.04).

**Deux constantes en unités absolues étaient silencieusement détunées par ce ×25** :
`GRAD_NORM_MAX = 1.0` (le clip mordait sur tous les batches) et le clamp α `[1e-5, 1e-2]`
(entropie réduite à 0.04 % de la loss actor).

Corrections appliquées le 2026-08-17 (branche `develop`) : γ 0.997 → 0.99 et **suppression de la
normalisation du reward** dans `SacReplayBuffer` (les rewards bruts sont déjà calibrés à la main
dans [−1, +2] ; diviser par l'écart-type d'un signal sparse amplifie au lieu de normaliser).

**How to apply:** sur la run suivante, attendre Q (donc `π_μ`) de l'ordre de l'unité, pas 15.
Juger sur `hit/fire` et `s_μ` (cf. ppo-reward-redesign-2026-08). Pistes non retenues à ce jour :
timer de famine en observation (proprioception-archi-boundary, refusé), buffer plus grand
(300 k / 32 tanks = seulement ~11 épisodes, ~0.02 % de transitions "hit"), famine en troncation
plutôt qu'en terminal.
