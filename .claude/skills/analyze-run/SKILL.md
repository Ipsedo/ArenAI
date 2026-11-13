---
name: analyze-run
description: Analyse le metrics.csv d'un dossier d'entraînement ArenAI (outputs/train_NNN_*) en le replaçant dans l'état du code au moment du run (commit effectif, diff vs run précédent) et dans sa config.json, puis écrit <run>/analysis.md. À utiliser dès qu'il faut lire un run ou son CSV (« analyse 504 », « où en est le run », « regarde le CSV de train_503 », « qu'est-ce qui s'est passé dans ce run »).
---

# Analyse d'un run

Entrée : un dossier `outputs/train_NNN_<algo>/` ou juste son numéro (`504`).
Sortie : `<run>/analysis.md` (en français) + `<run>/analysis_summary.json`, et un résumé dans le chat.

L'analyse n'est jamais une décision de conception : elle établit des faits, propose des options,
Samuel tranche (règle de CLAUDE.md).

## 1. Résumé chiffré

Depuis la racine du repo :

```shell
python3 .claude/skills/analyze-run/scripts/analyze_run.py <run> --json [--prev <run_de_référence>]
```

Le script (stdlib seule) affiche : datation (mtime de `config.json`, jamais celle du dossier,
qui peut être réutilisé), état en cours/fini, débit en pas/h et tank-steps/h, commit au lancement
et commits pendant le run, diff de config et diff de code **hors commentaires/blancs** avec le run
précédent, jalons D, moyennes par niveau de D, 10 tranches d'index, extrêmes, signaux
automatiques. Les signaux sont des pistes, pas des verdicts.

## 2. Établir le code exact du run

1. **Commit effectif.** Samuel lance souvent un run avant de commiter : le script marque les
   commits ≤ 90 min après le lancement (et avant le run suivant) comme « peut contenir le code du
   run » et en déduit un commit effectif estimé. Le confirmer :
   - les valeurs de `config.json` doivent coïncider avec les défauts de l'en-tête
     `arenai_agent/src/agents/<algo>/*_hyperparams.h` au commit effectif
     (`git show <hash>:<chemin>`), sauf options CLI explicites ;
   - lire la fiche mémoire du run (`run-NNN-*.md`) et celles des runs voisins : elles disent
     « non commité », « relancé », « binaire du HH:MM »… Si mémoire et git divergent, le dire.
   - run en cours avec un arbre de travail sale (`git status`) : comparer la mtime du binaire
     (`cmake-build-release/arenai_agent/arenai_agent_train` ou `build/…`) à la date de lancement.
2. **Lire le code au commit effectif, pas l'arbre de travail** (`git show <hash>:<chemin>`) pour
   tout ce que `config.json` ne contient pas :
   - reward et shaping : `arenai_agent/src/core/train_environment.cpp` (échelles hit/reçu,
     potentiel de visée, respawn/troncature à réserve vide) ;
   - événements, économie d'obus : `arenai_model/src/tank/jolt_enemy_tank.cpp`,
     `arenai_model/include/arenai_model/constants.h`, `shell_economy.h` ;
   - σ, entropie, α : `arenai_agent/src/networks/constants.h`, `networks/misc.cpp`,
     `networks/entropy.cpp`, `agents/entropy_alphas.cpp` ;
   - trainer de l'algo (`agents/<algo>/*_trainer.cpp`), curriculum (`core/spawn_curriculum.cpp`),
     définition des métriques (constructeurs `MeanMetric("<nom>", fenêtre, …)`).
3. **Ce qui change vs le run précédent** : lire le diff réel des fichiers listés par le script
   (`git diff <from> <to> -- <fichier>`), pas seulement les messages de commit. Nommer le ou les
   changements testés ; s'il y en a plusieurs, ils sont **confondus** — le dire.

## 3. Lire le CSV

Une ligne toutes les `30 / wanted_frequency` itérations ; `index` = itérations de la boucle
d'entraînement = décisions (période `action_repeat × wanted_frequency`, 0,1 s par défaut). La
ligne `index = 0` est ignorée. `save_N` ≈ `N × save_every`. Les colonnes changent d'un run à
l'autre : se fier à l'en-tête réel et, pour une colonne inconnue ou suspecte, à sa définition au
commit effectif.

| colonne | sens (au 11/11/2026, à revérifier au commit du run) |
|---|---|
| `r`, `r_hit`, `r_rcv`, `r_aim` | reward moyen par tank-décision, fenêtre très longue (4 × nb_tanks × max_episode_steps) ⟹ lissé, en retard |
| `miss` | distance de ratage moyenne (m) des obus retombés |
| `s` | durée de vie (s) des tanks morts |
| `fire` | tirs/s par tank vivant, fenêtre 256 décisions ; **aveugle au dump** de début d'épisode |
| `hit` | touches/tirs par épisode, fenêtre 16 épisodes (≠ hit/obus de la sonde) |
| `kill` | kills par épisode, fenêtre 16 épisodes |
| `side`, `D` | côté de spawn joué / borne haute du curriculum |
| `π`, `v` | pertes acteur / critic |
| `ev` | variance expliquée **in-sample** (tous minibatchs de tous les epochs) : plus d'epochs critic la gonfle |
| `Hc` | entropie continue moyenne des dims (masque une dim saturée) |
| `Hd` | entropie discrète ; fire masqué hors états armés ⟹ ≈ H(zoom) |
| `clip`, `kl`, `skip` | fraction clippée, KL par rollout, fraction de minibatchs sautés par `target_kl` |
| `alpha_*` | température d'entropie par dimension d'action (CSV seulement) |

## 4. Interpréter

- Lire `MEMORY.md` et les fiches liées (runs précédents de la même lignée, feedbacks). Ne
  **jamais** re-proposer un levier marqué refusé (cooldown long, contrainte de cadence, tête
  auxiliaire, duplication vision → proprio, …).
- Comparer à **D égal**, pas à index égal ; bruit de seed : jalon D ±30 k, réserve ×2, un record
  isolé ne prouve rien. Seuls sont structurels les effets imposés par le code.
- Débits à nb_tanks différent : comparer en tank-steps ou en temps mur.
- Régimes connus à reconnaître : régime KL (skip ↑, kl ≥ target), collapse de σ (Hc ↓ < 0),
  critic faible (ev < 0,2 ; ev précède hit de 10-20 k), gâchette éteinte (fire → 0, kill → 0),
  dump (invisible au CSV : décomposition du reward ou sonde), μ aux bornes, α hors plancher.
- Le CSV ne dit rien de la sélectivité de la gâchette ni de la réserve vidée : si la question
  l'exige, proposer une sonde (`arenai_agent_probe`, step = 0,1 s) plutôt que conclure.
- Chaque affirmation chiffrée porte son index et son D.

## 5. Écrire `<run>/analysis.md`

En français, sans remplissage :

1. **En-tête** : run, algo, lancement → dernière ligne, en cours/fini, index final, débit,
   commit effectif (et degré de confiance).
2. **Ce que teste ce run** : diff config + code vs le run précédent, hypothèse (mémoire),
   changements confondus.
3. **Lecture du CSV** : jalons D, trajectoire par phase, chiffres clés à D égal avec les runs de
   référence connus en mémoire.
4. **Diagnostic** : mécanismes avec leurs preuves ; séparer établi / dans le bruit / non tranché.
5. **Options** (pas de décision) et **repères** : quoi regarder, à quel D/index, quelle sonde.

Puis un résumé court dans le chat avec le lien vers le fichier. Si l'analyse établit un fait
nouveau, mettre à jour la fiche mémoire du run (ou en créer une), en suivant les règles mémoire.
