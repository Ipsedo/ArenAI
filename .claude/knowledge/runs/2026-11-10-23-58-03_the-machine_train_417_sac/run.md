# train_417_sac

| | |
|---|---|
| machine | the-machine |
| utilisateur | samuel |
| lancement | 2026-11-10 23:58:03 |
| commit au lancement | `39ae97f9` — update probe for actor (no liquid) |
| commits ≤ 90 min après | `50a5e256`, `ff5fc4ab`, `40a6f467`, `11779aaf`, `3de97377` |
| diff au lancement | inconnu (identité relevée après la fin du run) |
| données brutes | `outputs/train_417_sac` sur the-machine |

## Faits établis

_SAC run 417 (2026-11-11, = 415 + replay stratifié 0.7 + plafond σ 0.7, target_sigma 0.1, buffer 600k) — pire que 415 et 416 : curriculum 2× plus lent (D=300 à 48k, 350 à 114k), pic hit 0.103, régression continue à D=350 (0.10 → 0.072, kill 3.8) ; sonde : tir inconditionnel, angle au tir 16.6° → 28.6°, corr(μ, erreur) ≈ 0_

Run 417 (relancé 2026-11-10 23:58, binaire du WIP devenu 50a5e256 ; layout d'état inchangé, vérifié),
analysé à 153k pas / 10 h, encore en cours. Diff de config vs 415 : `armed_fraction` 0.7 + buffer
600k (vs 800k) ; plafond σ 0.7 (e52e76ff) présent, absent de 415. Même `target_sigma` 0.1 que 415.

CSV, moyennes par palier (415 / 416 / 417) :
- D=250 : fin à 20k / 20k / **47k**.
- D=300 : hit 0.090 / 0.104 / 0.088, kill 4.3 / 5.6 / 3.3 ; 417 y reste 65k pas.
- D=350 : hit 0.117 / 0.103 / 0.086, kill 11.9 / 8.0 / 3.8, fire 1.54 / 0.79 / 0.67 obus/s.
Pic hit 0.103 à 100k, puis déclin monotone 0.100 → 0.072 et miss 22.7 → 28.4 m sur 40k pas à D
fixe (ce n'est pas un simple saut de palier). Hc −3.6 dès 36k (comme 415), Q ≈ −0.2, perte q 0.027
(×1.5 vs 415), αd0 monte à 6e-3 à 100k puis redescend.

Sonde (D=350, 2 épisodes, échantillonnée) save_3 (81k) / save_5 (135k) :
- p(fire | can_fire) 0.86 / 0.88, quasi indépendant de l'angle ; à sec 84 / 88 % des décisions.
- σ canon 0.53 / 0.59 (415 : 0.42–0.50 ; plafond rarement atteint), |μ| site 0.69 → 0.98, saturation
  site 15 → 28 % : entropie toujours satisfaite par saturation.
- angle au tir médian 16.6° → 28.6°, part des obus > 30° 24 → 48 % ; hit à angle égal ~stable
  (< 2° : 0.19/0.17 vs 0.31 dans 415) → la régression = alignement qui se dégrade.
- corr(μ canon, erreur yaw/pitch) −0.05 / −0.02 : toujours aucune visée.

**Why:** attribution impossible : 416 (stratifié + plafond) sortait de D=250 à 20k comme 415, donc
la stratification n'explique pas la lenteur propre à 417 ; le refactor 50a5e256 relu = équivalent ;
aucune graine fixée (pas de manual_seed) → écarts 415/416/417 possiblement du bruit de seed. Seul fait
robuste : corr(μ, erreur) ≈ 0 dans les 3 runs, les réglages ne bougent le hit que de ±0.02.

**How to apply:** ne plus attribuer un écart entre runs SAC uniques à un hyperparamètre. Check proposé
(minutes, pas de run) : sur les états de la sonde, corr(∂Q/∂a_canon, erreur yaw/pitch) → le critique
voit-il la valeur de viser ? (nécessite d'étendre la sonde, à valider par Samuel).
Voir run 415, run 416, sac-agent-2026-11.

2026-11-11, gradient du critique (sonde étendue : colonnes `dq_canon_x/y` = ∂E[min Q]/∂a_canon au mode,
save_5, D=350, non commité) : Δyaw = +7.4°·act par décision (corr 0.42) ; **P(∂Q/∂a dans le sens qui
réduit |yaw|) = 0.47–0.50 dans toutes les tranches**, y compris armé + can_fire + |yaw| < 5° ; corr ≈ 0.
|∂Q/∂a| ~6e-4 (1.4e-3 armé). → le critique ne voit pas la valeur de viser : verrou = crédit, pas acteur.
Site non interprétable : Δpitch ne répond pas à act_canon_y (corr 0.00), géométrie `pitch` à vérifier.
Pas de bug de facteur 3 sur « secondes avant réserve vide » : 1 pas de sonde = 3 frames = 0.1 s,
cooldown 5 frames → 1 tir / 2 décisions, 10 obus en ≥ 2 s (minimum physique).
