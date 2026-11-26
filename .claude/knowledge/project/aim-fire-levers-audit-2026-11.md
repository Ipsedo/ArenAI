---
name: aim-fire-levers-audit-2026-11
description: "Audit complet 2026-11-02 de la chaîne vision→action→reward→PPO pour expliquer pourquoi visée/tir n'apprennent pas ; chiffres clés et leviers structurels identifiés"
metadata:
  node_type: memory
  type: project
  originSessionId: 2d01f42b-282f-4b64-acf4-9b156cf0b517
  modified: 2026-11-02T08:04:45.262Z
---

Audit du 2026-11-02 (branche develop, 171db7fb). Chiffres mesurés dans le code/assets :
- Spawn D=250, 32 tanks : ennemi le plus proche médian 22 m (p10 8 m, p90 42 m) → tank 54×23 px
  dans l'image 256×128 (FOV 45° vertical / 79° horizontal), tolérance lacet ±8°. La résolution
  et le bruit σ=0.3 (5.4°/décision) ne sont PAS le verrou à D=250.
- Terrain heightmap6 : 18.6 km × 9 km, hauteurs 0–200 m ; dans la fenêtre de 250 m au centre :
  35 m de relief, pente moyenne 12° → ligne de vue souvent coupée, élévation requise ±13°, la
  reward de visée vise « l'ennemi le plus proche » même derrière une colline.
- Couleur aléatoire par item par épisode (9 pièces d'un tank = 9 couleurs), 120 formes
  distractrices 2.5–10 m de même distribution : l'ennemi n'est identifiable que par la forme.
- Économie du tir asymétrique : famine = truncation bootstrapée (gratuite), réserve vide =
  terminaison −0.1, coût 0.01/obus → ne pas tirer est l'optimum au plancher aléatoire (run 392).
- Horizon de crédit γλ ≈ 0.984 (~60 pas) pour un événement obus à 1–4 pas (vol 0.1–0.4 s à
  D=250) ; le meilleur run (388) avait λ 0.95.
- Vérifié sans bug : synchro tir/observation (producteurs appelés avant Update → l'obus part
  dans l'orientation observée), GAE/truncation/masque fire identiques dans les 3 trainers,
  maths PPO. Bug mineur : collision obus-obus comptée hit/kill (rare).
- Aucun plafond mesuré : pas d'oracle scripté ni de sonde perception supervisée.

**Why:** Samuel trouvait que les réglages d'hyperparamètres successifs étaient vains ; l'audit
pointe des verrous structurels (économie du tir, crédit, identification visuelle, terrain,
absence de plafond mesuré) plutôt que des réglages.

**How to apply:** avant tout nouveau run, proposer l'oracle privilégié + sonde perception pour
mesurer le plafond ; traiter l'asymétrie famine/réserve vide comme décision de design à
valider ; voir [[fire-cost-per-shell]], [[out-of-shells-termination]],
[[ppg-train-392-diagnosis]], [[ppo-train-388-diagnosis]].
