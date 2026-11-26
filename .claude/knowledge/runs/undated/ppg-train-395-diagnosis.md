---
name: ppg-train-395-diagnosis
description: "Run 395 (PPG, aim Cauchy, sans coût de tir) — hit 0.077 → 0.03 verrouillé sur D 250→300 alors que r monte ; cause = économie à point de bascule (échange de paire ≤ 0), pas Cauchy ni les hyper-paramètres PPG ; suite = hit +0.3 / reçu -0.2"
metadata:
  node_type: memory
  type: project
  originSessionId: 22b3c1ad-5726-43a8-bbb0-26231322ea86
  modified: 2026-11-02T21:07:38.502Z
---

Run 395 PPG (2026-11-02, binaire ef514d06 = aim reward Cauchy 1/(1+(miss/tol)²), tol 2.5 m + 3°·d,
plus de coût de tir, out_of_shells -0.5, γ 0.99 / λ 0.95). Même config que 393/394.

- À D=250 : hit 0.055 → 0.078, kill 0 → 1.1. Itération 59 (53k) : D 250 → 300, hit 0.077 → 0.05
  en 4 itérations, courte reprise à 0.06 (66k), puis 0.03 et kill = 0 à 89k. 394 a absorbé le même
  palier (0.054 → 0.07, puis D=350) ; hit (fenêtre 16 épisodes) bruité ±0.015 à D constant.
- Décomposition r : r_aim, r_hit **et** r_rcv baissent ensemble, morts moins fréquentes, s 15 → 20 s,
  fire 0.8 → 0.45, miss constant 20 m, **r total monte**. L'agent optimise bien : la population
  converge vers l'arène calme, par cadence/précision (392 l'avait fait par fire → 0).
- **Cauchy n'est pas la cause** (première lecture erronée, corrigée par Samuel) : le ppo_liquid
  sur l'autre PC tourne avec Cauchy et monte à D=400. Cauchy garde une pente loin de la cible
  (utile pour apprendre la visée) et paie un peu plus les obus qui ne touchent pas : il rend le
  problème confortable, il ne le crée pas.
- Cause retenue = économie à point de bascule : un hit = +0.2 (+aim ≤ 0.1) pour le tireur, -0.3
  pour la cible → échange de paire ≤ 0, seul le kill (+2/-1, 5 coups) est positif ; une vie = 10
  obus puis out_of_shells -0.5 (s ≈ 17 s ≈ 10 obus à 0.6/s), que tirer lentement repousse hors de
  l'horizon γ (10 s). Engager n'est rentable que si p(hit) reste au-dessus d'un seuil ; 395 l'a
  franchi au palier D=300, 394 est resté juste au-dessus, le liquid (mémoire) tient.
- Hyper-paramètres PPG hors de cause : Hc 0.21 → 0.10 = σ 0.30 → 0.27 ; clip 0.2-0.25, kl
  0.013-0.016 < 0.05, skip 0, identiques à 393/394. Note : l'aux distille sur le seul dernier
  rollout, pas les n_pi accumulés.
- **Décision 2026-11-02 (Samuel)** : échange positif pour la paire, hit +0.3 / reçu -0.2
  (jolt_enemy_tank.cpp, tests alignés : mort + reçu = -1.2).

**Why:** Samuel a tranché contre ma lecture Cauchy avec le contre-exemple liquid ; ne pas conclure
d'une comparaison à un run par variante.
**How to apply:** juger sur hit/fire + décomposition r_rcv/mort/s, pas sur r. Voir
fire-cost-per-shell, [[ppg-train-392-diagnosis]], out-of-shells-termination,
aim-fire-levers-audit-2026-11.
