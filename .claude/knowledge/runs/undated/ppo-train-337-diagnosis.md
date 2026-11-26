---
name: ppo-train-337-diagnosis
description: "train_337 ppo_liquid skip+CNN large (2026-09-28) — updates trop violents (clip 0.33, KL > cible, 25-45 % d'epochs coupées), hit en recul après 700k ; meilleurs checkpoints save_2/3"
metadata:
  type: project
---

Run analysé le 2026-09-28 (commit 1604dc5c : skip Linear+LayerNorm+SiLU en parallèle du LTC
chez l'acteur et le critique, CNN 6 blocs 8→256 canaux, neuron_number 512, epochs 4,
minibatch 1024, truncated normal, commande de visée en **delta** — l'absolu a été backtracké
le 2026-09-24 par bb66d985, cf. aim-absolute-position-control). 1.21M steps en 14.4 h
(23 steps/s contre 32 pour train_329 : réseau ~27 % plus lent).

Constat : `clip` 0.30-0.37 et `kl` 0.05-0.065 (> target_kl 0.05) dès les 100k premiers pas,
`skip` 25-45 % — contre clip ≈ 0.05 / kl ≈ 0.005 / skip ≈ 0 pour train_329 (liquid, epochs 2,
minibatch 512) et train_336 (PPO vanilla truncated normal, epochs 1). hit/s monte à 0.08
(200-300k) puis recule à 0.03-0.05 après 800k, kill 8 → 2, alors que 329 tient 0.09 à D=350-400.
`ev` instable (plongeons à −1.3, −0.5). Hc moyen ≈ 0 avec α_c > 0 → au moins une dim sous sa
cible (probablement déplacement, −0.2). Hd 0.61 → 0.41 (cible 0.28).

Lecture : le pas de politique par rollout est ~10× trop grand (epochs 4 + réseau plus gros +
branche skip feed-forward non lissée par le LTC), la garde KL coupe mais trop tard.
Pistes proposées (non validées) : epochs 2, ou lr acteur réduit, ou ablation skip vs CNN.
Meilleurs checkpoints : save_2 / save_3 (216k / 324k). Voir ppo-liquid-agent,
[[ppo-train-326-diagnosis]].
