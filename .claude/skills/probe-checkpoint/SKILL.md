---
name: probe-checkpoint
description: Lance la sonde (arenai_agent_probe) sur un checkpoint ArenAI (outputs/train_NNN_*/save_K) après avoir vérifié que le binaire correspond au code du run, puis analyse ses métriques (gâchette vs alignement, hit/obus, réserve vidée, tourelle, σ/μ) et écrit <run>/probes/save_K.md. À utiliser pour « sonde save_9 de 504 », « lance la sonde sur ce checkpoint », « est-ce que la gâchette est sélective à save_12 ».
---

# Sonde d'un checkpoint

Entrée : `outputs/train_NNN_<algo>/save_K` (ou un run + `--save K`), éventuellement les côtés de
spawn, le nombre d'épisodes, le mode déterministe.
Sortie : `<run>/probes/save_K_side<S>[_det].{csv,_shots.csv,log,json}`, `<run>/probes/save_K.md`
(en français) et un résumé dans le chat.

Comme les autres skills d'analyse : établir des faits, proposer, ne rien décider.

## 1. Compatibilité binaire ↔ checkpoint (avant tout lancement)

```shell
python3 .claude/skills/probe-checkpoint/scripts/probe_checkpoint.py <save> --check-only
```

La sonde rejoue le checkpoint **avec le code compilé dans le binaire**, pas avec celui du run :
réseau, distributions, observation et environnement (respawn à réserve vide, économie…) viennent
du binaire. Le script prend le binaire le plus récent parmi `cmake-build-release/` et `build/`
(`--binary` pour en imposer un), le date (mtime → commit) et affiche :
- la tête σ/μ du checkpoint (`actor_repr.txt` : `Clamp` vs `SoftClamp`, `mu` Linear vs
  Sequential…) ;
- le diff hors commentaires entre le commit effectif du run et celui du binaire sur le code
  d'inférence, et les modifs non commitées antérieures au build.

S'il y a un diff, le script refuse de lancer. Lire le diff : s'il change la forme ou
l'interprétation d'un tenseur (paramétrage de σ, couche `mu`, taille ou ordre de
l'observation), la sonde est **fausse**. Le chargement peut quand même réussir, comme pour
`Clamp`/`SoftClamp`. Il faut alors un binaire au bon commit (worktree + skill `build`, après
accord de Samuel), ou une conversion du checkpoint (`arenai_agent/python/convert_mu_linear.py`
sur une copie). Si le diff est sans effet sur l'inférence (refactor, environnement inchangé),
relancer avec `--force` et le dire dans l'analyse.

## 2. Lancer

```shell
python3 .claude/skills/probe-checkpoint/scripts/probe_checkpoint.py <save> [--sides 550,1000] [--episodes 4] [--deterministic]
```

Défauts : côtés 550 et 1000, 4 épisodes, stochastique, `--cuda`. L'économie d'obus
(`shells`, `shells_per_hit`, `max_shells`) vient de `config.json` du run, pas des défauts du jeu.
Compter ~40 s par épisode et par côté à 32 tanks (≈ 5-6 min avec les défauts). Lancer en arrière-plan (`run_in_background`) et attendre la
notification. La sonde partage le GPU avec un entraînement en cours. `--analyze-only` réanalyse
les CSV existants sans relancer.

## 3. Métriques (décisions armées = `can_fire` et réserve > 0, step ≥ 10)

Un step = une décision = `action_repeat × wanted_frequency` = 0,1 s par défaut. Les valeurs en
secondes notées en mémoire avant le 11/11 sont 3× trop grandes.

| métrique | sens |
|---|---|
| `ratio`, `p_al`, `p_off` | p_fire armé aligné (< 5° **et** cible < 40 m) / hors axe (≥ 20°) : la **sélectivité** de la gâchette, définition des records en mémoire depuis le 10/11. `ratio_angle_only` (< 5° seul) est dans le JSON |
| `p_arm` | niveau moyen de la gâchette ; plat vs angle = pas de sélectivité |
| déterministe (p > 0,5) | ce que fait le jeu en mode déterministe : une gâchette sous 0,5 partout est muette en jeu |
| `hit/ob`, `h<5°`, `tirs<5°`, `enrich` | touches par obus retombé, par obus tiré aligné, part des tirs alignés, enrichissement (part des tirs alignés / part des décisions alignées) |
| `vide_s`, `%vident`, `%désarmé` | 1re réserve vide (médiane, s), part des tank-épisodes qui vident, part des pas désarmés : le **dump** |
| `tour<5°` | part des décisions où la tourelle est à < 5° d'une cible : la visée |
| `σx`, `σy`, `σ@plaf`, `|μ|cx`, `μsat` | σ canon médian, part au plafond (≥ 0,69), \|μ\| canon médian, part \|μ\| > 1,5 (saturation tanh) |
| `p_zoom`, corr(p_fire, p_zoom) | couplage gâchette/zoom |

Regarder les effectifs (`n` aligné & proche) : sous ~300, le ratio est bruité.

## 4. Interpréter

- Situer le checkpoint : index = K × `save_every`, D du curriculum à cet index, CSV autour
  (skill `analyze-run` si `<run>/analysis.md` manque ou est périmée).
- Comparer aux repères en mémoire **au même D et au même côté** (`metriques-croisiere-d1000-*`,
  fiches des runs), en tenant compte du bruit de seed. Un record isolé n'établit rien.
- Côté 550 = compétence à portée moyenne, côté 1000 = conditions de croisière. Lire les deux.
- Pièges connus : gâchette plate vs angle à un niveau quelconque = pas d'apprentissage ;
  sélectivité brute faussée par le réflexe de spawn, d'où step ≥ 10 ; « tourelle cassée » en jeu
  = souvent la réserve vide ; Hd du CSV ≈ entropie du zoom, donc la gâchette se lit ici
  seulement.

## 5. Écrire `<run>/probes/save_K.md`

En-tête (run, save, index, D, binaire + commit, compatibilité, côtés/épisodes/mode), table de
résumé, lecture (gâchette, tir, dump, visée, têtes), comparaison aux repères au même D,
conclusions établies vs bruit, sondes ou saves suivantes utiles. Résumé court dans le chat. Si un
fait nouveau est établi, mettre à jour la fiche mémoire du run.
