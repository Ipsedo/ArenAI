import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arenai_run import (  # noqa: E402
    KEY_COLUMNS, alpha_columns, build_context, code_diff, config_diff, fmt, fmt_value, load_run,
    previous_run, summary, table,
)


def k(index):
    return f"{index / 1e3:.0f}k"


def main():
    parser = argparse.ArgumentParser(description="Résumé chiffré d'un run ArenAI")
    parser.add_argument("run", help="dossier du run, ou son numéro (504)")
    parser.add_argument("--prev", help="run de référence pour les diffs (défaut : numéro précédent)")
    parser.add_argument("--json", action="store_true", help="écrit <run>/analysis_summary.json")
    args = parser.parse_args()

    run = load_run(args.run)
    ctx = build_context(run)
    s = summary(run)
    cols = [c for c in KEY_COLUMNS if c in run.columns]
    alphas = alpha_columns(run)

    print(f"# {run.name}")
    print(f"lancé {ctx['launch']} (mtime config.json), dernière écriture CSV {ctx['last_write']}"
          f"{' — EN COURS' if ctx['running'] else ''}")
    print(f"{s['nb_rows']} lignes, index final {k(s['last_index'])}, {ctx['hours']} h, "
          f"{ctx['steps_per_hour']} pas/h, {ctx['tank_steps_per_hour']} tank-steps/h")

    print("\n## Code")
    c = ctx["commit_at_launch"]
    print(f"commit au lancement : {c['hash']} {c['date']} {c['subject']}" if c else "aucun commit")
    e = ctx["effective_commit"]
    if e and c and e["hash"] != c["hash"]:
        print(f"commit effectif estimé : {e['hash']} (à confirmer contre config.json / mémoire)")
    for cm in ctx["commits_during_run"]:
        tag = "  <- peut contenir le code du run (lancé non commité)" if cm["maybe_run_code"] else ""
        print(f"  pendant le run +{cm['minutes_after_launch']} min : {cm['hash']} {cm['subject']}{tag}")

    prev_path = args.prev or previous_run(run)
    if not prev_path:
        print("\n## Différences : aucun run précédent dans outputs/")
    else:
        prev = load_run(prev_path)
        build_context(prev)
        print(f"\n## Différences vs {prev.name}")
        diff = config_diff(prev, run)
        print("config.json :" if diff else "config.json : identique")
        for key, (a, b) in diff.items():
            print(f"  {key}: {fmt_value(a)} -> {fmt_value(b)}")
        cd = code_diff(prev, run)
        if cd:
            print(f"code {cd['from']}..{cd['to']} :")
            for line in cd["log"]:
                print(f"  {line}")
            print("\n".join(f"  {line}" for line in cd["stat"].splitlines())
                  or "  aucun changement hors commentaires")
            if cd["comment_only_files"]:
                print(f"  (+ {cd['comment_only_files']} fichiers modifiés seulement en commentaires/blancs)")
        added = [c for c in run.columns if c not in prev.columns]
        removed = [c for c in prev.columns if c not in run.columns]
        if added or removed:
            print(f"colonnes CSV : +{added} -{removed}")

    print("\n## Config (agent / environment)")
    for section in ("agent", "environment"):
        items = run.config.get(section, {})
        print(f"{section}: " + ", ".join(f"{key}={fmt_value(v)}" for key, v in items.items()))

    print("\n## Jalons D (index de première apparition)")
    print("  " + ", ".join(f"D {d:.0f} @ {k(i)}" for d, i in s["d_milestones"]))

    print("\n## Moyennes par niveau de D")
    show = ["n", "from", "to"] + cols
    print(table(["D"] + show, [[d] + [st[c] for c in show] for d, st in s["per_d_level"].items()]))

    print("\n## Fenêtres (10 tranches d'index)")
    show = ["to", "D"] + cols
    print(table(show, [[w[c] for c in show] for w in s["windows"]]))
    if alphas:
        print()
        show = ["to"] + alphas
        print(table(["to"] + [a.replace("alpha_", "α_") for a in alphas],
                    [[w[c] for c in show] for w in s["windows"]], width=10))

    print("\n## Extrêmes (valeur @ index, D)")
    for c, e in s["extremes"].items():
        mx, mn, last = e["max"], e["min"], e["last"]
        print(f"  {c:>14}: max {fmt(mx[0]).strip()} @ {k(mx[1])} D{fmt(mx[2]).strip()}"
              f" | min {fmt(mn[0]).strip()} @ {k(mn[1])} | dernier {fmt(last[0]).strip()}")

    print("\n## Signaux automatiques (à vérifier, pas des verdicts)")
    for sig in s["signals"] or ["aucun"]:
        print(f"  - {sig}")

    if args.json:
        out = run.path / "analysis_summary.json"
        out.write_text(json.dumps(s, indent=1, default=str))
        print(f"\nécrit : {out}")


if __name__ == "__main__":
    main()
