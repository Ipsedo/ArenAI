# train_421_ppg

| | |
|---|---|
| machine | the-machine |
| utilisateur | samuel |
| lancement | 2026-11-11 22:10:42 |
| commit au lancement | `c02137c6` — absolute angle for turret |
| commits ≤ 90 min après | aucun |
| diff au lancement | inconnu (identité relevée après la fin du run) |
| données brutes | `outputs/train_421_ppg` sur the-machine |

## Faits établis

_Run 421 PPG (c02137c6, visée en consigne absolue yaw cos/sin + pitch, MIN_ALPHA 1e-4, vision 256×128) — D=400 à 145k puis bloqué ; hit/fire plat 0.086 → 0.077, kill ×2 par cadence (fire 1.1 → 2.1/s), Hc → −1.0 sans gain_

Run 421 PPG lancé 2026-11-11 22:10 sur c02137c6 (confiance haute), analysé à 515k / 9.2 h (en cours).
Changements confondus : consigne absolue de visée (5 actions continues, lacet atan2(sin, cos) dans le
repère châssis, pas limité), cibles d'entropie [0, 0, −1, −1, −1], MIN_ALPHA 1e-3 → 1e-4, gros
réseau/vision 256×128. Premier PPG depuis la refonte reward du 06/11 → références = SAC 415-420.

- D : 300 @ 21k, 350 @ 114k, 400 @ 145k puis bloqué. Pic hit 0.103 @ 133k (D=350).
- À D=400 : hit 0.086 → 0.077, miss 31 → 36 m, mais kill 7 → 19 et r ↑ par **volume** (fire 1.1 → 2.1/s).
- Hc 0.23 → −1.0 ; α tous au plancher 1e-4 jusqu'à ~200k (dir) / ~440k (visée) → contraction libre, sans gain.
- skip ~0.4 et clip ~0.3 depuis 200k ; ev 0.9 → 0.4.
- Caméra montée sur le canon : en consigne absolue l'acteur doit ajouter l'angle tourelle (proprio).

**Why:** 1er test de la consigne absolue depuis le revert du 24/09 ([[aim-absolute-position-control]]).
**How to apply:** ne pas lire kill/r comme progrès ; options proposées : sonde save_5/save_19,
arrêt, ou rejouer 421 en delta pour isoler la consigne. Voir [[sac-train-415-diagnosis]].
