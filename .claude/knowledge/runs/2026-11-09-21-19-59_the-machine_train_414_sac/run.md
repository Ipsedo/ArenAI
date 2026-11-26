# train_414_sac

| | |
|---|---|
| machine | the-machine |
| utilisateur | samuel |
| lancement | 2026-11-09 21:19:59 |
| commit au lancement | `2e594069` — clamp alpha to 0.001 |
| commits ≤ 90 min après | aucun |
| diff au lancement | inconnu (identité relevée après la fin du run) |
| données brutes | `outputs/train_414_sac` sur the-machine |

## Faits établis

_SAC run 414 (2026-11-09/10) — premier SAC au-dessus du plancher aléatoire ; verrou levé = décroissance d'α jusqu'à ~5e-4 (45k pas perdus) ; pic hit 0.117 à D=300 (save_3) puis régression à 0.067 avec Q surestimé +0.4_

Run 414 (config : réseaux 128/64, batch 512, gpu_batch 1024, gradient_steps 2, α₀ 0.01,
lr α 3e-5, target_sigma 0.1, fire/zoom proba init 0.5, 32 tanks, 162k pas en 10h40,
27k pas ≈ 1h47). Saves tous les 27k : save_3 = 81k = meilleur.

Trois phases :
1. 0–45k : Hc 2.7→2.1 (σ≈0.9), hit 0.04–0.05 = plancher aléatoire, kill 1→0.1. α décroît
   à la vitesse max d'Adam (3e-5/update en log) : 0.01 → 6e-4. **Rien n'apprend tant que
   α > ~5e-4** ; 412 (gradient_steps 1) n'avait atteint que 2.6e-3 à 45k → même plancher.
2. 45k–80k : dès α ≈ 5e-4, le terme Q domine, Hc s'effondre 2.1 → −3.5 (cible σ 0.1 atteinte
   à ~78k, Hc cible −3.53), hit 0.05 → 0.117, kill → 5, curriculum 250 → 300 (68k).
3. 80k–162k : α_c ≈ 7e-5 (à la cible, sans régularisation effective), hit 0.117 → 0.094 à
   D=300 constant, puis 350 (115k) → 0.067 ; kill 6.8 → 2.7 ; miss 25 → 28 m ; fire monte
   0.5 → 1.0 (118–133k) ; perte critique q ×6 (0.004 → 0.026) ; Q moyen 0.33–0.43 alors que
   le retour réalisé par vie ≈ 0 (reward zero-sum, r ≈ −0.0003/pas) → **surestimation ≈ +0.4**.

**Why:** la durée de la phase morte est purement le temps de décroissance d'α (lr α × nb
updates), pas un problème d'exploration ; la régression ensuite ressemble à la dérive classique
SAC sans entropie (politique quasi déterministe exploitant les erreurs du critique).

**How to apply:** pistes à valider avec Samuel, pas décidées : α₀ ≈ 5e-4 ou lr α ×10 pour
sauter la phase morte ; séparer régression et curriculum (D figé à 300) ; régulariser le critique
(target_sigma 0.2, gradient_steps 4, clip Q) ; tester save_3 en jeu. Voir
[[sac-train-412-diagnosis]], sac-agent-2026-11.
