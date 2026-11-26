---
name: out-of-shells-termination
description: "2026-10-30 réserve d'obus vide = fin d'épisode réelle à -0.3, abaissée à -0.1 le 2026-10-31 ; famine reste une truncation"
metadata:
  node_type: memory
  type: project
  originSessionId: d6038644-cb02-48ba-a1f0-2f1494ead829
  modified: 2026-10-30T11:42:59.936Z
---

2026-10-30 : à la demande de Samuel, la mort par réserve vide (`out_of_shells`) redevient une terminaison avec pénalité -0.3 (au lieu de -1) ; plus de bootstrap. La famine (`starved`) reste une truncation sans pénalité.

**Why:** choix de Samuel sur la récompense ; remplace la truncation out_of_shells lue dans [[ppo-train-371-diagnosis]] et [[ppo-train-378-diagnosis]].

**How to apply:** `is_timeout()` = famine seule ; `is_suicide()` inclut `out_of_shells`, pour que ces morts ne comptent pas dans `nb_kills_episode`. Les runs antérieurs ne sont pas comparables sur r et ev.

2026-10-31 : pénalité abaissée à -0.1 (run 382 : un tir coûtait ~0.03 pour ~0.014 de hit espéré, l agent apprenait à ne plus tirer).
