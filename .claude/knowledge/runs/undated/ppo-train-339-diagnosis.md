---
name: ppo-train-339-diagnosis
description: "train_339 = 337 avec epochs 2 (5.5M steps, 2026-09-30) — KL/skip réglés mais clip 0.15-0.2 (3× 329) ; curriculum bloqué à D=700 dès 3.6M ; hit plafonne ~0.08 ; ev instable ; effondrement transitoire à 4.3M"
metadata:
  type: project
---

Run analysé le 2026-09-30 (autre PC de Samuel, 31 steps/s, 49 h). Config = train_337 à
l'identique sauf `epochs 2` (train_338 = même config, arrêté à 578k). Cibles Hc
`{-0.2,-0.2,-0.88,-0.88}` (visée en delta), facteurs discrets 0.4.

- Epochs 2 a fait son travail : kl 0.015-0.025 (< target 0.05), skip ≈ 0-2 % (max 9 %),
  contre 0.06 / 25-45 % pour 337. Mais clip reste 0.15-0.24 (≈ 0.10-0.15 en fin), 3-4× 329.
- Pic hit/s 0.095-0.098 à 600-900k (D 300-350, kill ~11), puis ~0.08 ; D monte jusqu'à 700
  à 3.6M et y reste 2M steps (probes sous ratio_high 0.06) → plateau de compétence.
- ev : moyennes de tranche souvent négatives (−2.5 à 200k, −2.1 à 900k, −7 à 4.2M) ; 329 était
  plus stable. Effondrement 4.26→4.35M (hit 0.075→0.033, kill 8→1.5) sans pic de clip/kl,
  récupéré vers 4.5M.
- Hc tombe 0.45→0.09 en 200k (338 : 0.48→0.24, 329 : 0.60→0.36) ; α_c > 0 en permanence →
  une dim sous plancher (probablement déplacement). Hd 0.62→0.34 (cible 0.277).

Lecture : le problème KL de 337 est résolu, le plafond vient d'ailleurs (critique instable,
Hc qui s'effondre tôt). Meilleurs checkpoints probables : save_6/7/8 (≈650-860k).
Voir [[ppo-train-337-diagnosis]], ppo-liquid-agent.
