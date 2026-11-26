---
name: sac-agent-2026-11
description: "Agent SAC recodé le 2026-11-08 (sous-commande `sac`) — politique hybride factorisée, espérance exacte sur les combos discrets, 6 α (un par action, validé 2026-11-08), lr α 3e-5, replay uint8 en anneau par tank"
metadata:
  node_type: memory
  type: project
  originSessionId: 71808214-6837-4246-aa7a-b04852f6abb7
  modified: 2026-11-08T14:01:44.330Z
---

Le 2026-11-08 un agent `sac` a été ajouté dans `arenai_agent/src/agents/sac/` (+ `networks/sac_actor`, `networks/q_function`),
sur le design validé par Samuel en discussion : tanh-gaussienne reparamétrée (continu), Bernoulli (fire, zoom) avec
**espérance exacte sur les 2^D combinaisons** (critique à K sorties, style SAC-Discrete), **un α par action** (4 axes continus puis fire, zoom ; validé par Samuel
le 2026-11-08 après le run 410 où 2 α globaux laissaient un axe payer l'entropie d'un autre), cible par axe H(tanh-gauss σ=0.3)
≈ 0.128 nat et par Bernoulli 0.3·ln2 ; métriques `αc0..3`, `αd0..1`, Hc/Hd = sommes ; masque de tir appliqué aux probas (p·mask) et exclu de l'entropie, replay **uint8 stocké une fois**
(ring par tank, s' = slot suivant, `finish_episode` écrit un slot observation seule), done ∨ truncated = cut sans bootstrap.
Batch d'optim + bloc GPU (`--batch_size`, `--gpu_batch_size`) comme PPO.

**Why:** Samuel veut tester un off-policy après les plateaux PPO ; il réduira lui-même le CNN (3 couches) et l'image à 128×64
pour maximiser le buffer en RAM CPU (≈25 KB/transition → 1 M ≈ 25 Go).

**How to apply:** points que j'ai fixés seul et à faire valider : σ **dépendant de l'état** (standard SAC, à l'inverse du
log_sigma libre des agents PPO), défauts `initial_alpha` **0.01 depuis le 2026-11-08 (0.1 au run 411, entropie ≫ reward)**, `target_sigma` 0.3 (validé 2026-11-08 : ≈ 5° lacet / 2° site par décision, à baisser quand D > 400), `alpha_learning_rate` 3e-5 (÷10 le 2026-11-08, run 410 : α ÷85 en 11k pas), `discrete_target_entropy_ratio` 0.3,
`grad_norm_max` 10, `tau` 0.005, pas de n-step/PER. Intégration jeu (enum public `AgentAlgorithm`, menu desktop) **non faite** :
contrat `include/` → demander d'abord. Voir [[sac-train-293-diagnosis]] et [[sac-train-309-diagnosis]] pour les échecs
SAC historiques (α effondré, buffer de 10 épisodes).

2026-11-10 : commité sur develop (5d47c0d6 + a15ee8f1), rebasé sur origin. Les α du SAC vivent dans
`networks/sac_alpha.h` (`SacAlpha`, **non borné**) ; `AlphaParameters` (PPO/PPG) est clampé à [1e-3, 1] et ne
doit pas être réutilisé pour SAC. α₀ par défaut = 5e-4 (run 415). Les 6 α sont écrits dans le CSV seulement.

2026-11-10 (suite run 415) : **échantillonnage stratifié du replay** validé par Samuel (option a) :
`--armed_fraction` (défaut 0.7) = part de chaque batch tirée parmi les transitions où la réserve
d'obus > 0 (`SHELLS_PROPRIOCEPTION_INDEX` = ENEMY_PROPRIOCEPTION_SIZE − 2, défini dans
`networks/constants.h`), le reste parmi les transitions à sec ; strate vide → l'autre prend tout.
Choisi plutôt que de rendre la réserve vide terminale pour ne pas impacter les agents PPO/PPG.

2026-11-11 : **retours n-step** (`--n_step`, défaut 10, validé par Samuel) : le replay calcule
G = Σγᵏr sur m ≤ n pas (arrêt à coupure / fin d'épisode / slot le plus récent) et
`bootstrap_discounts` = γᵐ (0 si coupé), qui remplace `cuts` ; γ vit dans le replay, plus dans
SacTrainer. Bonus d'entropie intermédiaires ignorés, pas de correction hors-politique. Raison :
hit = 8 décisions après le tir (q90 16), le TD 1-pas ne transmettait pas le crédit (∂Q/∂a aléatoire,
run 417). n = 200 (20 s) refusé : biais + variance. Non commité au 2026-11-11.

2026-11-11 : encodeur d'état partagé (style SAC-AE/DrQ, sans augmentation) : `StateEncoder` (vision +
capteurs) unique, porté par TwinQNetwork (une seule instance pour Q1/Q2, entraînée par la perte TD) ;
`SacActor` (dans actor.h) lit l'encodage détaché, optimiseur sur `head_parameters()` ; encodeur cible
suit le Polyak τ. `Actor` (PPO/PPG) inchangé. Checkpoints SAC antérieurs incompatibles. Validé par
Samuel : vision + capteurs, un encodeur pour les jumeaux, pas d'augmentation.
