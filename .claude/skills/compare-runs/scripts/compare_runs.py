import argparse
import math
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "analyze-run" / "scripts"))

from arenai_run import (  # noqa: E402
    KEY_COLUMNS, alpha_columns, at_index, build_context, code_diff, config_diff, d_milestones,
    flatten, fmt, fmt_value, load_run, per_d_level, signals, table,
)


def k(index):
    return "-" if index is None else f"{index / 1e3:.0f}k"


def short(run):
    return str(run.number)


def main():
    parser = argparse.ArgumentParser(description="Comparaison chiffrée de plusieurs runs ArenAI")
    parser.add_argument("runs", nargs="+", help="dossiers ou numéros de runs (ordre = référence d'abord)")
    parser.add_argument("--step", type=float, default=100e3, help="pas des points à index égal")
    parser.add_argument("--half-width", type=float, default=15e3, help="demi-fenêtre autour de chaque point")
    parser.add_argument("--columns", help="colonnes à comparer, séparées par des virgules")
    args = parser.parse_args()

    runs = [load_run(r) for r in args.runs]
    for r in runs:
        build_context(r)

    common = [c for c in runs[0].columns if all(c in r.columns for r in runs)]
    cols = args.columns.split(",") if args.columns else [c for c in KEY_COLUMNS if c in common]

    print("# Comparaison " + " vs ".join(r.name for r in runs))

    print("\n## Runs")
    for r in runs:
        ctx = r.context
        e = ctx["effective_commit"]
        print(f"{r.name}: lancé {ctx['launch']}, {len(r.rows)} lignes, index final "
              f"{k(r.rows[-1]['index'] if r.rows else 0)}, {ctx['steps_per_hour']} pas/h, "
              f"{ctx['tank_steps_per_hour']} tank-steps/h{', EN COURS' if ctx['running'] else ''}")
        if e:
            print(f"  code : {e['hash']} {e['date']} {e['subject']}")
        if (e and ctx["commit_at_launch"] and e["hash"] != ctx["commit_at_launch"]["hash"]):
            print(f"  (commit au lancement {ctx['commit_at_launch']['hash']}, "
                  f"commit effectif estimé {e['hash']} : à confirmer)")

    print("\n## config.json : clés qui diffèrent")
    keys = set()
    for r in runs[1:]:
        keys |= set(config_diff(runs[0], r))
    if not keys:
        print("identique")
    else:
        flats = [flatten(r.config) for r in runs]
        print(table(["clé"] + [short(r) for r in runs],
                    [[key] + [fmt_value(f.get(key)) for f in flats] for key in sorted(keys)],
                    width=12))

    print("\n## Code entre runs consécutifs (dans l'ordre donné)")
    for a, b in zip(runs, runs[1:]):
        cd = code_diff(a, b)
        print(f"{a.name} -> {b.name} : {cd.get('from')}..{cd.get('to')}")
        for line in cd.get("log", []) or ["  aucun commit"]:
            print(f"  {line}")
        if cd:
            print("\n".join(f"  {line}" for line in cd["stat"].splitlines())
                  or "  aucun changement hors commentaires")
            if cd["comment_only_files"]:
                print(f"  (+ {cd['comment_only_files']} fichiers modifiés seulement en commentaires/blancs)")

    print("\n## Colonnes CSV")
    for r in runs:
        missing = [c for c in runs[0].columns if c not in r.columns]
        extra = [c for c in r.columns if c not in runs[0].columns]
        if missing or extra:
            print(f"{r.name}: -{missing} +{extra}")
    print(f"communes comparées : {', '.join(cols)}")

    print("\n## Jalons D (index de première apparition)")
    miles = [dict(d_milestones(r)) for r in runs]
    levels = sorted(set().union(*miles))
    print(table(["D"] + [short(r) for r in runs], [[d] + [k(m.get(d)) for m in miles] for d in levels]))

    print("\n## À D égal (moyennes sur toutes les lignes du niveau)")
    per_d = [per_d_level(r, cols + alpha_columns(r)) for r in runs]
    for c in ["n"] + cols:
        rows = []
        for d in levels:
            vals = [p.get(d, {}).get(c) for p in per_d]
            if all(v is None for v in vals):
                continue
            rows.append([d] + vals)
        print(f"\n{c}")
        print(table(["D"] + [short(r) for r in runs], rows))

    print(f"\n## À index égal (moyenne ±{k(args.half_width)} autour de chaque point)")
    last = min(r.rows[-1]["index"] for r in runs if r.rows)
    points = [args.step * i for i in range(1, int(last // args.step) + 1)] or [last]
    for c in ["D"] + cols:
        rows = []
        for p in points:
            vals = []
            for r in runs:
                st = at_index(r, cols, p, args.half_width)
                vals.append(None if st is None else st.get(c))
            rows.append([k(p)] + vals)
        print(f"\n{c}")
        print(table(["index"] + [short(r) for r in runs], rows))

    print("\n## Pics (valeur @ index, D)")
    for c in cols:
        cells = []
        for r in runs:
            pts = [(x[c], x["index"], x.get("D")) for x in r.rows if c in x and not math.isnan(x[c])]
            if not pts:
                cells.append(f"{short(r)}: -")
                continue
            v, i, d = max(pts)
            cells.append(f"{short(r)}: {fmt(v).strip()} @ {k(i)} D{fmt(d).strip()}")
        print(f"  {c:>6} max | " + " | ".join(cells))

    print("\n## Signaux automatiques par run (à vérifier, pas des verdicts)")
    for r in runs:
        print(f"{r.name}:")
        for sig in signals(r) or ["aucun"]:
            print(f"  - {sig}")

    print("\n## Analyses écrites existantes")
    for r in runs:
        md = r.path / "analysis.md"
        if md.exists():
            stale = md.stat().st_mtime < (r.path / "metrics.csv").stat().st_mtime
            print(f"{md}{' (antérieure à la dernière ligne du CSV)' if stale else ''}")
        else:
            print(f"{r.name}: pas d'analysis.md")


if __name__ == "__main__":
    main()
