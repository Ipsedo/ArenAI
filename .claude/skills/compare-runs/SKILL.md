---
name: compare-runs
description: Compare plusieurs runs d'entraînement ArenAI (outputs/train_NNN_*) à partir de leurs analyses de CSV, en tenant compte des différences de code (commits effectifs, diff hors commentaires) et de config.json entre les runs, puis écrit outputs/comparisons/<date>_<runs>.md. À utiliser pour « compare 502 503 504 », « 504 vs 502 », « est-ce que le changement X a servi », « lequel est le meilleur ».
---

# Comparaison de runs

Entrée : deux runs ou plus (dossiers ou numéros), **la référence en premier**, éventuellement la
question posée (« le SoftClamp a-t-il servi ? »).
Sortie : `outputs/comparisons/<AAAA-MM-JJ>_<n1>_vs_<n2>[_vs_…].md` (en français) + un résumé dans
le chat.

Comme pour `analyze-run` : établir des faits et proposer, ne rien décider à la place de Samuel.

## 1. Une analyse à jour par run

Pour chaque run, si `<run>/analysis.md` manque ou est antérieure à la dernière ligne de
`metrics.csv` (le script de comparaison le signale), appliquer d'abord le skill `analyze-run`
à ce run. Lire ensuite chaque `analysis.md` : elles portent le commit effectif confirmé, ce que
teste chaque run et les pièges déjà identifiés.

## 2. Tables comparatives

Depuis la racine du repo :

```shell
python3 .claude/skills/compare-runs/scripts/compare_runs.py <ref> <run2> [<run3> …] [--step 100000] [--half-width 15000] [--columns hit,kill,ev,Hc]
```

Le script réutilise `analyze-run/scripts/arenai_run.py` et affiche : état et débit de chaque run,
commit effectif estimé, clés de `config.json` qui diffèrent, diff de code hors commentaires entre
runs consécutifs (dans l'ordre donné), colonnes CSV absentes, jalons D, moyennes **à D égal**,
moyennes à index égal, pics, signaux, et l'état des `analysis.md`.

## 3. Inventorier toutes les différences

Une comparaison ne vaut que si l'on sait ce qui diffère. Lister, pour chaque paire :
- **config.json** : chaque clé qui diffère (nb_tanks, rollout, epochs, lr, cibles d'entropie…) ;
- **code** : lire le diff réel des fichiers listés entre commits effectifs
  (`git diff <a> <b> -- <fichier>`) ; reward, économie, σ/entropie, trainer, curriculum d'abord ;
- **hors git** : code non commité au lancement, binaire différent, dossier relancé (fiches mémoire
  et `analysis.md`) ;
- **métriques** : une colonne dont la définition a changé entre les commits
  (`git log -S'"<nom>"' <a>..<b>`) n'est pas comparable telle quelle — le dire.

Plus d'une différence ⟹ effets **confondus** : ne pas attribuer l'écart à l'une d'elles.

## 4. Comparer honnêtement

- **À D égal d'abord** (même stade du curriculum), index égal ensuite ; tank-steps ou temps mur
  si nb_tanks diffère. Ne pas comparer au-delà du D atteint par le run le moins avancé ; signaler
  les runs en cours.
- **Bruit de seed** (n = 1 par config) : jalon D ±30 k, réserve vidée ×2, hit/obus ±1 point entre
  configs identiques. Exiger un écart tenu sur plusieurs niveaux de D ou plusieurs saves. Un
  calendrier D identique entre runs n'est pas une preuve d'équivalence (curriculum seedé).
- **Structurel vs bruit** : un effet imposé mécaniquement par le code (σ jamais au plafond avec
  une borne douce, α qui monte quand Hc passe sous la cible…) est établi ; un écart de niveau de
  hit/kill n'est établi que s'il dépasse le bruit.
- `ev` in-sample (gonflé par plus d'epochs critic), `fire` aveugle au dump, `Hd` ≈ H(zoom) :
  ne pas comparer ces colonnes naïvement entre configs qui changent ces mécanismes.
- Lire `MEMORY.md` et les fiches des runs : ne pas re-proposer un levier refusé.

## 5. Écrire la comparaison

`outputs/comparisons/<AAAA-MM-JJ>_<n1>_vs_<n2>….md` :

1. **Runs** : état, index/D atteints, commit effectif, débit.
2. **Différences** : config, code (résumé du diff réel), hors git, métriques ; changements confondus.
3. **Tableau à D égal** des métriques qui répondent à la question (pas toutes les colonnes).
4. **Trajectoires** : jalons D, pics, dérives (où et à quel D chaque run décroche).
5. **Verdict** par question : établi / dans le bruit / non tranché, avec les chiffres.
6. **Pour trancher** : ce qu'il faudrait (D à atteindre, sonde, run de contrôle isolant une
   variable) — options, pas décisions.

Puis un résumé court dans le chat avec le lien vers le fichier.
