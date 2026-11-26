# train_423_ppg

| | |
|---|---|
| machine | the-machine |
| utilisateur | samuel |
| lancement | 2026-11-12 23:46:38 |
| commit au lancement | `7422d098` — skip separately between policy action groups (canon skip and prevent fire action to learn) |
| commits ≤ 90 min après | aucun |
| diff au lancement | inconnu (identité relevée après la fin du run) |
| données brutes | `outputs/train_423_ppg` sur the-machine |

## Faits établis

_Run 423 PPG (802b1b32 : gun laying + avantage visée φ sans critique + arrêt KL par groupe) — 1er run à D=1000 (429k), hit 0.126 @ D400 → 0.19-0.22 @ D1000, fire 1.6 → 0.8/s, ev 0.74, Hc ~0 ; 422 = même config arrêt KL joint, stoppé à 148k_

Run 423 PPG lancé 2026-11-12 23:46 sur 802b1b32 (confiance haute), analysé à 581k (en cours).
Testé vs 422 : arrêt KL par groupe (visée canon vs reste), 7422d098. Hérité de 422 (1e171bbf) :
stabilisation canon, avantage canon = potentiel φ trajectoire, aim_gae_lambda 0, visée en delta.

- D : 500 @ 145k, 1000 @ 429k (borne finale, jamais atteinte avant ; 421 bloqué à 400).
- hit 0.126 (D400) → 0.193 (D1000), 0.217 @ 581k ; fire 1.6 → 0.8/s ; kill plat ~21 ; miss ∝ D.
- ev 0.70-0.75, Hc creux −1.3 @ 59k puis ~+0.1 ; α canon actifs 0.01-0.035.
- 422 (arrêt KL joint, stoppé à 148k) : mêmes jalons D et hit jusqu'à 145k, mais skip 0.5-0.6 et Hc −1.05 ;
  423 skip ≤ 0.15. Gain au-delà de 148k non attribuable entre les deux commits.

**Why:** premier vrai déblocage du curriculum, voir [[ppg-train-421-diagnosis]].
**How to apply:** curriculum saturé (hit ≫ 0.08) → juger sur hit/fire à D=1000 ; sélectivité de la
gâchette non prouvée par le CSV → sonde save_18-21 proposée.

**Sonde save_21 (567k, D=1000, 2026-11-13)** : gâchette sélective, ratio 4.9 (côté 550) / 7.5 (côté 1000),
hit/obus 0.23 / 0.19, désarmé 6-10 % (dump supprimé), tourelle < 5° 24-28 % ; mais corr(μ canon, erreur) ≈ 0
→ l'alignement ne vient pas d'une commande canon proportionnelle. Déterministe : tire dans 9-24 % des
états alignés seulement. La sonde gère désormais PPG (branche DistillationActor dans probe.cpp).
