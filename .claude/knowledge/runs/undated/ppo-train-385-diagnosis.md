---
name: ppo-train-385-diagnosis
description: "Run 385 (384 + aim reward dense réactivée par Samuel, commit a185f109) — à 133k steps identique à 384 : r_aim plat à 0.005/step (= baseline aléatoire, décroît avec D), miss 22 → 24 m, hit 0.06, μ uniforme sur les 4 axes ; l'oracle à l'échelle 0.1/obus ne ressort pas du bruit d'avantage"
metadata:
  type: project
---

Run `outputs/train_385_ppo` (2026-10-31 19h37) = 384 + `detail.aim` remis dans le retour de
`get_reward` (commit a185f109 de Samuel, qui a aussi commité σ libre / lr 1e-4 / σ₀ 0.05 /
entropie 0.001 / réserve vide −0.1). Aim reward = 0.1 · exp(−miss / (10 m + 6°·distance)),
créditée à l'impact de chaque obus (≈ 0.054 par obus à 22 m de miss à 250 m).

- À 133k steps : r_aim 0.0054 → 0.0042 par step, jamais au-dessus de sa valeur sous politique
  aléatoire (0.0044 au step 0) ; elle baisse quand D monte. miss 22 → 24 m, hit 0.06-0.07,
  kill 1.5 → 0.1, curriculum 250 → 300 (32k) → 350 (129k) sur le plancher aléatoire, comme 384.
- μ : normes 0.07 sur les 4 axes à save_1 (108k), identiques déplacement/visée, comme 384.
- σ tenu (Hc −1.58 → −1.64), skip ≤ 0.03, clip 0.3, ev 0.25-0.3, Hd 0.69 → 0.63.
- Ordre de grandeur : ∂r/∂miss ≈ 0.0015 par mètre et par obus ; les retours sont dominés par
  death −1, received −0.3, réserve vide −0.1 → l'aim reward est invisible dans l'avantage
  (cf. [[ppo-train-373-diagnosis]] : aim ×2 avait déjà échoué).

**Why:** l'oracle « aim reward telle quelle » ne discrimine pas : trop faible pour dominer la
variance des retours. Il reste deux hypothèses : rapport signal/bruit de l'avantage, ou perception
(cible 5×2 px à 250 m, [[ppo-train-345-diagnosis]]).

**How to apply:** pour trancher, un run diagnostic où le signal de visée domine tout (aim ×10
et death/received neutralisés) ou une sonde perception (D fixe 100 m sans curriculum). Choix
de conception à valider par Samuel.
