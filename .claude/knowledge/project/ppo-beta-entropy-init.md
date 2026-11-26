---
name: ppo-beta-entropy-init
description: PPO Beta — historique (μ, κ·300) ; 2026-09-09 : ppo_liquid en mode/κ Kruschke, tête κ log-échelle [2.01, 2000] ; 2026-09-23 : cibles de visée rebarémées pour la commande en position
metadata:
  node_type: memory
  type: project
  originSessionId: 89c6ac18-a0cb-49e4-bb33-cae34477cd95
  modified: 2026-09-23T00:00:00.000Z
---

Symptôme (2026-08-30) : avec la loi Beta dans PPO, l'agent tirait bien mais ne se déplaçait presque pas. Cause : `initial_sigma = 0.242` → α=β≈8, cloche serrée autour de 0 (H≈0 nat/action), et `continuous_target_entropy_init = 0` n'exerçait aucune pression vers le haut — alors que le discret (tir) démarrait à 0.98×H_max. On-policy → jamais de données de déplacement.

Décision Samuel : `initial_sigma = 0.5` (α=β=1.5, H≈0.645 vs ln 2≈0.693 pour l'uniforme) et `continuous_target_entropy_init = 0.6`. L'uniforme exact (σ=0.577) a été écarté : il exige un biais softplus ≈ −9.2 où le gradient des têtes α/β vaut ~1e-4 (têtes quasi gelées).

**Pourquoi :** l'exploration continue doit être quasi uniforme au départ pour que l'agent découvre la valeur du déplacement ; la paramétrisation `1 + softplus` impose de rester un peu au-dessus de α=β=1 pour garder des gradients sains.

**How to apply :** si l'agent se re-fige en « sniper statique » malgré cette init, le suspect suivant est le reward (cf. [[sac-train-309-diagnosis]], optimum local spray statique), pas l'entropie. Voir aussi [[ppo-reward-redesign-2026-08]].

Mise à jour 2026-09-01 : le backtrack Beta→truncated normal (216bb41f) a été annulé à la demande de Samuel — la Beta (`1 + softplus`, init unimodale depuis σ) est de nouveau la distribution continue de PPO et SAC. Même session : le PID Lagrangien projette désormais α sur [0, MAX_ALPHA] (vraie contrainte d'inégalité H ≥ cible, λ ≥ 0) — l'alpha signé « équality constraint » de 4a40a991 est un choix révoqué, ne pas le re-proposer.

Mise à jour 2026-09-02 : la Beta est reparamétrée en (μ, κ) — α = μ·κ·KAPPA_SCALE, β = (1−μ)·κ·KAPPA_SCALE, têtes μ/κ en `RangeSigmoidOutput`. Décisions Samuel : `KAPPA_SCALE = 300` ; plancher de la tête κ `MIN_KAPPA_SIGMOID = 2/KAPPA_SCALE` (= l'uniforme α=β=1, atteignable asymptotiquement, jamais de α<1 à μ=0.5) ; `initial_concentration = 0.0075` (α=β≈1.13, H≈0.686, quasi uniforme — le plancher exact donnerait un logit infini) ; `continuous_target_entropy = −0.88` = H de la Beta sur [−1,1] à α=β=49.5, soit σ d'action ≈ 0.1. La cible est par action (broadcast dans le PID). Repères : H max = ln 2 ≈ 0.693, H = 0 vers α=β=8 (σ≈0.24), H min atteignable ≈ −1.43 (κ max = 1 → α=β=150, σ≈0.058). L'ancien plancher H=0.6 et σ_init=0.5 sont obsolètes. Bug corrigé au passage : `init_kappa_output_weights` inversait la sigmoïde avec min=EPSILON au lieu du plancher réel de la tête ; le biais μ (logit) doit être 0, pas 0.5.

Mise à jour 2026-09-09 (test local, **non validé**) : re-test de la Beta sur **ppo_liquid uniquement** (le PPO simple reste en truncated normal), reparamétrée mode/κ à la Kruschke : α = ω(κ−2)+1, β = (1−ω)(κ−2)+1 (déjà dans `beta_law.cpp`, garantit α,β ≥ 1 dès κ ≥ 2). Contexte littérature : le standard publié est α,β = softplus(x)+1 (Chou ICML 2017, Hsu 2020, CaRL 2025, RLlib) ; mode+κ n'a pas de précédent RL publié mais impose la même contrainte d'unimodalité. Choix Samuel : tête κ en **échelle log** (softplus asymptotiquement linéaire = mauvais aux grands κ ; en log, pas d'entropie ≈ constant, comme le log_std gaussien). Implémentation : `ConcentrationOutput` = 2 + exp(log-interp sigmoïde entre CONCENTRATION_MIN=2.01 et MAX=2000) ; tête mode = Sigmoid, biais 0 (centré) ; κ₀ dérivé de `initial_sigma` (σ action sur [−1,1]) via κ = 1/σ²−1 (0.5 → κ₀=3), clampé aux bornes (le 0.f de la factory jeu → clamp à MAX, poids rechargés ensuite de toute façon). Action déterministe = mean (CaRL : mean > mode). Les vieux checkpoints ppo_liquid (têtes mu/sigma) ne se rechargent plus. Même session : `continuous_target_entropy` de ppo_liquid passé de 0.4 (σ≈0.36) à **−0.88** (σ≈0.1, κ équilibre ≈ 99) — Samuel voulait une policy plus déterministe, valeur alignée sur celle validée le 2026-09-02 ; `initial_sigma = 0.5` conservé (exploration initiale, l'équilibre ne dépend que de la cible).

Mise à jour 2026-09-23 : le passage de la visée en **consigne absolue** ([[aim-absolute-position-control]]) invalide le barème des deux dims de visée — les dims de déplacement (−0.2) ne bougent pas. La cible d'entropie est un **plancher** (α clampé à [0, MAX_ALPHA] dans `entropy.cpp`, λ ≥ 0) : elle interdit d'être plus piqué que la valeur donnée, donc elle interdit désormais de **viser** fin, là où avant elle interdisait seulement une *vitesse* fine.

Conversion à retenir : pour une entropie différentielle `h` sur [−1,1], largeur effective `e^h` et **σ_action ≈ e^h/√12**. Avec h = −0.88 → σ ≈ 0.12, soit 0.72°/frame de bruit de vitesse avant, mais **±21.6° de consigne de lacet** après (0.12 × 180°). Effet aggravant : en vitesse l'intégrateur moyennait le bruit d'action, en position il atterrit directement sur la visée — et l'obus part sur l'action échantillonnée.

Nouvelles valeurs (Samuel, 2026-09-23) : **`{-0.2, -0.2, -3.0, -1.4}`** par défaut dans `ppo_hyperparams.h` et `liquid_ppo_hyperparams.h`. Fourchette dérivée : égaler la granularité par frame d'avant donne h ≈ −4.3, égaler la dispersion accumulée sur ~30 frames donne h ≈ −2.6 ; −3.0 est le milieu. **Les deux dims de visée ne méritent plus le même chiffre** : le lacet couvre 180° et le site 36°, donc à précision physique égale `h_site ≈ h_lacet + ln 5 ≈ +1.61` (l'asymétrie existait déjà mais ne valait que ln(1/0.4) ≈ 0.92).

**How to apply :** ce sont des ordres de grandeur, à confirmer en lisant `Hc` sur les premiers milliers de pas (cf. [[ppo-train-326-diagnosis]] : lire Hc/Hd pour prédire le comportement en jeu). Si la visée reste grossière, resserrer le lacet avant de toucher au reste. Ne pas proposer d'annealing d'entropie, refusé (cf. [[ppo-reward-redesign-2026-08]]).
