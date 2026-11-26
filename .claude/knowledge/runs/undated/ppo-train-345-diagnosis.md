---
name: ppo-train-345-diagnosis
description: "train_345 (ppo_liquid, σ indépendant de l'état, entropie fixe 0.01, BigVisionNetwork 256x128) — analysé 2026-10-18 à 11.9M steps (en cours) : pas de régression, curriculum 250→850 à kill ~4.3 constant, mais σ monte 0.5→0.8/0.68 et kl ~3e-4"
metadata:
  node_type: memory
  type: project
  originSessionId: a79b48f0-2f23-42e5-b781-e29a3fb48841
  modified: 2026-10-18T14:47:19.849Z
---

Run lancé 2026-10-16 19:02, analysé 2026-10-18 à 11.9M steps (toujours en cours, save_110).
Config = suite validée de [[ppo-train-344-diagnosis]] : `sigma_logit` indépendant de l'état,
`continuous/discrete_entropy_factor = 0.01`, minibatch 2048, epochs 2, lr 1e-4, vision 256x128,
**BigVisionNetwork + CNN 8-16-32-48-64 non commités** (working tree au lancement).

- Meilleur run du curriculum : D monte 250 → 850 sans jamais redescendre, kill reste ~4.3-4.6
  et hit ~0.075-0.083 à chaque palier (344 : kill 2.9 à D=550 puis régression). Plateau de D
  le plus long à 700 (5.5-9.2M).
- σ (lu dans actor.pt via torch.jit.load) monte de façon monotone : déplacement 0.5 → 0.80/0.77,
  visée 0.5 → 0.68/0.69 (save_110), croissance qui ralentit. Hc 0.556 → 0.627 (max ln2 = 0.693).
  L'entropie 0.01 domine le gradient de σ ; la visée échantillonnée reste très bruitée.
- Têtes μ / discrète : normes de poids croissent (μ visée lacet 0.01 → 0.53) → l'apprentissage
  passe par la moyenne ; biais discrets inchangés (0.41/0.26), Hd ≈ 0.68.
- kl 1e-3 → 3e-4, clip 1 % → 0.1 %, skip 0 : mises à jour minuscules (σ large ⇒ ratio peu sensible
  à Δμ). ev 0.25-0.45 avec pics négatifs isolés. fire ≈ 1.1/s et s ≈ 35 constants.

Pistes proposées (non validées) : laisser finir ; tester save récent en jeu (mode μ ≠ politique
échantillonnée, cf. [[ppo-train-326-diagnosis]]) ; baisser seulement `continuous_entropy_factor`
(0.001-0.003) au prochain run ; lr plus haut si on veut des kl ~0.01.

Suivi 2026-10-19 à 16.9M (save_156, toujours en cours, ~73 steps/s) : D bloqué à 850 de 10.0 à
15.7M (≈ 5.7M steps) puis 900 ; kill 4.2 / hit 0.073 stables, aucune régression. σ continue de
monter, plus lentement : déplacement 0.86/0.83, visée 0.71/0.74 ; Hc plafonne ~0.63. kl 2-3e-4,
clip ~0.1 %. Lecture : plateau d'apprentissage, progrès du curriculum ralenti ×3 environ.

Suivi 2026-10-20 à 22.4M (save_207, en cours) : D=900 depuis 15.7M (6.7M steps). Le curriculum
monte si hit/fire des épisodes sondes (spawn à D max, 20 %) > 0.06 → à 900 l'agent est pile au
seuil. hit moyen 0.076, kill 4.05 → 4.7 (19-21M) → 4.4. σ visée lacet quasi figé (0.732 → 0.734),
déplacement 0.90/0.87 ; norme μ visée lacet 0.66 → 0.77 (save_185 → 207). kl ~2e-4, clip <0.1 %.

Suivi 2026-10-20 à 25.0M (en cours) : D=950 depuis ~24M ; hit 0.073-0.083 / kill ~4.3 plats
depuis 1M (précision constante pendant que D monte). kl 1.5e-4, clip 0.04 %, Hc 0.636.
Ordres de grandeur calculés : char ≈ 6.5 u de large → ≈ 1000/d px à 256x128 (FOV vertical 45°,
focale 154 px) ; tourelle π rad/s = 6°/frame → σ visée 0.73 ≈ 4.4°/frame de bruit vs demi-largeur
cible 1.9° à 100 u, 0.6° à 300 u. Analyse CNN (BigVisionNetwork) : 1re conv 5x5 s4 → 8 canaux
(compression ×12 là où vivent les cibles de 2-3 px), carte finale 4x2 aplatie = cellules de ~20°
→ localisation fine à coder dans les canaux. Pistes proposées (non validées) : entropie continue
0.001-0.003 d'abord ; spatial softmax (keypoints x,y) sur carte 16x8/32x16 au lieu du flatten ;
1re couche stride 2 + plus de canaux ; sonde linéaire « relèvement ennemi » sur encodeur gelé
pour prouver si le CNN est le goulot.
