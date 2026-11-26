# train_421_ppg — analyse (2026-11-12, run en cours)

## En-tête

- PPG, 32 tanks, vision 256×128, CNN 4 étages [16, 32, 64, 128], MLP 256/128.
- Lancé 2026-11-11 22:10 → dernière ligne 2026-11-12 07:23 (9.2 h), **en cours**, index 515k.
- Débit 55.9k pas/h (1.79 M tank-steps/h), ≈ 1.9× les SAC 419/420 (29.8k pas/h).
- Commit effectif : **c02137c6** « absolute angle for turret » (commit 22:05, binaire 22:10:07,
  run 22:10:42 ; config = défauts de `ppg_hyperparams.h` à ce commit). Confiance haute.
- Saves tous les 27k : save_K ≈ K × 27k (save_19 ≈ 513k).

## Ce que teste ce run

Premier PPG depuis 395 (2026-11-02) — entre-temps : refonte du reward du 06/11 (dégâts en fraction
de PV, potentiel de visée Cauchy, coût de tir 0.03, famine supprimée) et la lignée SAC 411-420.
La comparaison directe avec 395 n'est donc pas lisible ; les références utiles sont les SAC 415-420
(même reward).

Changements de c02137c6, **confondus** :

1. **Visée en consigne absolue** (re-tentative, cf. revert du 24/09) : 5 actions continues
   `dir_x, dir_y, yaw_cos, yaw_sin, pitch` au lieu de 4. Lacet = `atan2(yaw_sin, yaw_cos)` dans le
   repère du châssis, site = `pitch × MAX_ANGLE` ; dans les deux cas le pas est limité à
   `max_rad_per_frame` (`turret.cpp`, `canon.cpp`). Le joueur reste en delta.
2. Cibles d'entropie par dimension : `[0, 0, −1, −1, −1]` (avant : 0 pour toutes).
3. `AlphaParameters::MIN_ALPHA` 1e-3 → **1e-4** (touche PPO/PPG/liquid).
4. Plus gros réseau et vision 2× par rapport aux défauts SAC (pas de PPG récent pour dire ce qui
   a changé côté PPG).

## Lecture du CSV

Jalons D : 250 @ 1k, 300 @ 21k, 350 @ 114k, **400 @ 145k**, puis bloqué à 400 (370k décisions).

| phase (index) | D | hit | kill | fire/s | miss m | s | ev | Hc | Hd | skip |
|---|---|---|---|---|---|---|---|---|---|---|
| 0–20k | 250 | 0.076 | 3.3 | 1.24 | 21 | 68 | 0.69 | −0.03 | 0.69 | 0 |
| 21–113k | 300 | 0.088 | 5.7 | 1.13 | 23 | 63 | 0.71 | 0.09 | 0.65 | 0.003 |
| 114–144k | 350 | 0.101 | 6.8 | 1.09 | 26 | 72 | 0.73 | 0.19 | 0.52 | 0.006 |
| 145–206k | 400 | 0.086 | 9.1 | 1.41 | 31.5 | 61 | 0.70 | −0.22 | 0.55 | 0.14 |
| 206–360k | 400 | 0.083 | 14.2 | 1.69 | 32 | 60 | 0.66 | −0.51 | 0.56 | 0.44 |
| 360–515k | 400 | 0.077 | 16 | 1.75 | 33 | 61 | 0.62 | −0.72 | 0.39 | 0.37 |

- Pic hit 0.103 @ 133k (D=350). À D=400, hit glisse 0.086 → 0.077 en 370k décisions.
- kill 7 → 19 (max 19.7 @ 502k) et r_hit monte, **mais hit/fire ne monte pas** : la hausse vient
  du volume de tir (fire 1.1 → 2.1/s, max 2.9 @ 497k), pas de la précision. Miss 26 → 36 m à D égal.
- Hc 0.23 (140k) → −1.0 (515k), min −1.34 @ 501k ; Hd 0.69 → 0.35 (≈ H(zoom), fire masqué).
- α : tous au plancher 1e-4 de ~40k à ~200k. α_dir_x/y quittent le plancher à partir de ~220k
  (max 5.6e-3 / 1.4e-2 vers 365k) → l'entropie de déplacement est passée sous sa cible 0.
  α_yaw_cos/sin/pitch ne décollent qu'après ~440k (1.7e-3 / 6.6e-4 / 7.8e-4 à la fin) → l'entropie
  de visée vient d'atteindre −1. α_fire/zoom au plancher tout le run.
- Optimisation : kl par rollout 0.035 (< target 0.05) mais skip 0.35-0.55 depuis 200k et clip
  0.27-0.35 ; ev 0.9 (154k) → 0.40 en fin, v ×2-3.

## Diagnostic

**Établi**
- La consigne absolue ne débloque pas la visée dans ce run : à D=400, hit/fire plat puis en léger
  recul (0.086 → 0.077) et miss en hausse (31 → 36 m), alors que σ se resserre (Hc −0.2 → −1.0).
  C'est le motif connu « σ s'effondre sans gain de hit » (373, 383).
- La montée de kill et de r est un effet de cadence : plus d'obus à précision constante. Ne pas
  lire kill/r comme un progrès de visée.
- La régularisation d'entropie n'a pas freiné la contraction : α plancher abaissé à 1e-4 et cibles
  de visée à −1 → aucun coût d'entropie sur la visée jusqu'à ~440k. Le resserrement de la visée
  est donc libre, et il n'apporte rien au hit.
- Régime KL depuis 200k (skip ~0.4, clip ~0.3) : une bonne part des epochs acteur est coupée.

**Dans le bruit / non tranché**
- D=400 vs SAC 415 : hit 0.08 (421) vs 0.11 (415). Mais réseau, vision, algo et action changent
  ensemble ; impossible d'attribuer l'écart à la consigne absolue.
- Recul d'ev 0.9 → 0.4 : peut venir de la non-stationnarité (cadence ×2, plus de morts) plutôt que
  d'un critique dégradé ; pas de chute de hit corrélée 10-20k après.
- Le CSV ne dit pas si la visée suit la cible (corr μ/erreur) ni si la gâchette est sélective ou si
  la réserve est vidée : la hausse de fire peut être du dump.

**Point de mécanisme à garder en tête** : la consigne est dans le repère du châssis alors que la
vision voit depuis la tourelle ; pour viser, l'acteur doit ajouter l'angle tourelle courant
(proprioception) à l'écart vu à l'image. En delta, l'écart image suffisait.

## Options (décision Samuel)

1. Sonder save_5 (≈ 135k, pic hit à D=350) et save_19 (≈ 513k) : corr(μ_yaw/pitch, erreur),
   hit par obus selon l'alignement, p(fire | angle), réserve vide. Tranche si la visée absolue suit
   la cible et si la hausse de fire est sélective ou un dump.
2. Arrêter le run : 370k décisions à D=400 sans gain de hit/fire.
3. Pour isoler la consigne absolue : relancer à l'identique de 421 en delta (même réseau/vision/
   MIN_ALPHA) — sinon les quatre changements restent confondus.

## Repères

- hit/fire et miss à D=400, pas kill/r.
- Hc sous −1 avec α visée qui décolle : la cible −1 commence à agir ; si hit ne bouge pas d'ici
  ~600k, la contraction n'était pas le verrou.
- skip > 0.5 durable → régime KL qui gèle l'acteur.
