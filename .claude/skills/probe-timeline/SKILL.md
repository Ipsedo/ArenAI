---
name: probe-timeline
description: Analyse la chronologie des sondes d'un même entraînement ArenAI (<run>/probes/*.json, saves successives) en la croisant avec l'analyse du CSV, la config et le code du run, puis écrit <run>/probes/timeline.md. À utiliser pour « analyse les sondes de 504 », « comment évolue la gâchette au fil des saves », « la sélectivité tient-elle de save_13 à save_27 ».
---

# Chronologie des sondes d'un run

Entrée : un run (dossier ou numéro), éventuellement un sous-ensemble de saves.
Sortie : `<run>/probes/timeline.md` (en français) + un résumé dans le chat.

Établir des faits, proposer, ne rien décider à la place de Samuel.

## 1. Prérequis

- **Sondes** : les résultats du skill `probe-checkpoint` (`<run>/probes/save_K_side<S>.json`).
  Avec moins de deux saves sondées au même côté, il n'y a pas de chronologie. Proposer à
  Samuel les saves à sonder (aux changements de D, autour d'un pic ou d'un décrochage du CSV),
  sans les lancer d'office : c'est du temps GPU.
- **Analyse du CSV** : `<run>/analysis.md` du skill `analyze-run`. Si elle manque ou est
  antérieure à la dernière ligne de `metrics.csv`, appliquer d'abord `analyze-run`. Elle fournit
  le commit effectif, ce que teste le run, les phases et les régimes du CSV.

## 2. Tables

```shell
python3 .claude/skills/probe-timeline/scripts/probe_timeline.py <run> [--saves 5,9,13] [--half-width 15000]
```

Le script affiche :
- les binaires utilisés, avec un avertissement si la chronologie mélange des binaires ou contient
  une sonde faite malgré un diff d'inférence ;
- une table par côté et par mode (stochastique ou déterministe, nombre d'épisodes), une ligne par
  save, avec les métriques de `probe-checkpoint` et leurs effectifs ;
- les métriques du CSV autour de l'index de chaque save : D, hit, kill, fire, ev, Hc, Hd, kl,
  skip et l'α le plus haut ;
- les saves non sondées.

## 3. Analyser

1. **Comparabilité.** Ne mettre sur une même courbe que des sondes au même côté, au même mode,
   avec le même nombre d'épisodes et un code d'inférence équivalent. Si le binaire a changé en
   cours de chronologie, lire son diff (cf. `probe-checkpoint` §1) avant de comparer les saves
   de part et d'autre.
2. **Trajectoire par axe**, en séparant :
   - la gâchette : ratio, p_al, p_off, niveau, mode déterministe ;
   - le tir : hit/obus, part des tirs alignés, enrichissement ;
   - le dump : réserve vidée, % qui vident, % désarmé ;
   - la visée : tourelle < 5° ;
   - les têtes : σ, |μ|, saturation, zoom.

   Dire quel axe bouge en premier : par exemple la tourelle lâche avant la gâchette, ou la
   saturation de μ précède la perte de visée.
3. **Croiser avec le CSV et le code.**
   - Chaque changement de sonde a-t-il un pendant dans le CSV (hit, kill, ev, Hc, α, skip) au
     même index ? Ou le CSV y est-il aveugle (`fire` ne voit pas le dump, Hd ≈ zoom) ?
   - Un changement de D entre deux saves change la difficulté : un recul côté 550 n'est pas une
     régression si D a doublé. Regarder les deux côtés.
   - Rapporter chaque mécanisme au code du run : reward, économie, paramétrage de σ, α, ce que
     teste le run d'après `analysis.md`. Un effet imposé par le code est structurel ; le reste se
     juge contre le bruit.
4. **Bruit.** Le ratio est bruité sous ~300 décisions alignées & proches. 4 épisodes par sonde,
   n = 1 seed : exiger une tendance sur au moins 3 saves, ou un écart qui dépasse nettement la
   variabilité entre saves voisines. Les repères de mémoire (records, autres runs) se comparent
   au même D et au même côté.
5. **Mémoire.** Lire `MEMORY.md` et les fiches du run et de ses prédécesseurs. Ne pas
   reproposer un levier refusé.

## 4. Écrire `<run>/probes/timeline.md`

1. **Run et code** : commit effectif, ce que teste le run, binaire(s) de sonde.
2. **Table chronologique** des métriques clés par côté, avec D et le CSV au même index.
3. **Lecture par axe**, avec l'ordre des événements.
4. **Lien avec le CSV et le code** : mécanismes, avec leurs preuves.
5. **Établi / dans le bruit / non tranché.**
6. **Suite** : saves à sonder pour trancher, repères à surveiller. Ce sont des options, pas des
   décisions.

Puis un résumé court dans le chat. Si un fait nouveau est établi, mettre à jour la fiche mémoire
du run.
