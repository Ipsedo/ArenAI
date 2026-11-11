import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "analyze-run" / "scripts"))
sys.path.insert(0, str(HERE.parents[1] / "probe-checkpoint" / "scripts"))

from arenai_run import (  # noqa: E402
    alpha_columns, at_index, build_context, config_value, fmt, load_run,
)
from probe_metrics import HEADLINE  # noqa: E402

CSV_COLUMNS = ["hit", "kill", "fire", "ev", "Hc", "Hd", "kl", "skip"]


def main():
    parser = argparse.ArgumentParser(description="Chronologie des sondes d'un run ArenAI")
    parser.add_argument("run", help="dossier du run ou numéro")
    parser.add_argument("--saves", help="sous-ensemble de saves, ex. 5,9,13")
    parser.add_argument("--half-width", type=float, default=15e3,
                        help="demi-fenêtre CSV autour de l'index de chaque save")
    args = parser.parse_args()

    run = load_run(args.run)
    build_context(run)
    probe_dir = run.path / "probes"
    results = []
    for f in sorted(probe_dir.glob("*.json")):
        data = json.loads(f.read_text())
        if "meta" in data and "metrics" in data:
            results.append(data)
    if args.saves:
        wanted = {int(s) for s in args.saves.split(",")}
        results = [r for r in results if r["meta"]["save_index"] in wanted]
    if not results:
        raise SystemExit(f"aucune sonde dans {probe_dir} : utiliser le skill probe-checkpoint")

    groups = defaultdict(list)
    for r in results:
        m = r["meta"]
        groups[(m["side"], m["deterministic"], m["episodes"])].append(r)

    save_every = config_value(run, "save_every") or 0
    last_index = run.rows[-1]["index"] if run.rows else 0
    probed = sorted({r["meta"]["save_index"] for r in results})
    existing = sorted(int(p.name.split("_")[1]) for p in run.path.glob("save_*"))

    print(f"# Chronologie des sondes — {run.name}")
    ctx = run.context
    print(f"code du run : {ctx['effective_commit']['hash']} {ctx['effective_commit']['subject']}"
          if ctx["effective_commit"] else "code du run : inconnu")
    print(f"index final CSV {last_index / 1e3:.0f}k{' (EN COURS)' if ctx['running'] else ''}, "
          f"save_every {save_every}, saves sondées {probed}, saves existantes 0..{max(existing)}")

    print("\n## Binaires utilisés")
    seen = defaultdict(list)
    for r in results:
        m = r["meta"]
        key = (m.get("binary"), m.get("built"), (m.get("binary_commit") or {}).get("hash"))
        seen[key].append(f"{m['save']}@{m['side']:g}")
    for (binary, built, commit), saves in seen.items():
        print(f"{binary} (compilé {built}, {commit}) : {', '.join(saves)}")
    if len(seen) > 1:
        print("ATTENTION : plusieurs binaires — vérifier que le code d'inférence est le même")
    for r in results:
        if r["meta"].get("inference_diff"):
            print(f"ATTENTION : {r['meta']['save']}@{r['meta']['side']:g} sondé avec un diff "
                  f"d'inférence run → binaire")

    alphas = alpha_columns(run)
    for (side, det, episodes), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["meta"]["save_index"])
        print(f"\n## Côté {side:g}{' déterministe' if det else ''}, {episodes} épisodes")
        header = ["save", "index"] + [h for h, _ in HEADLINE]
        print(" ".join(f"{h:>8}" for h in header))
        for r in rs:
            m = r["meta"]
            print(" ".join([fmt(m["save"]), fmt(f"{m['index'] / 1e3:.0f}k")]
                           + [fmt(f(r["metrics"])) for _, f in HEADLINE]))
        print("\neffectifs (n aligné<5°&<40m / n hors axe / obus retombés)")
        for r in rs:
            t, s = r["metrics"]["trigger"], r["metrics"]["shots"]
            print(f"  {r['meta']['save']}: {t.get('n_aligned_close')} / {t['n_off_axis']} / "
                  f"{s.get('landed', 0):.0f}")

    print(f"\n## CSV autour de chaque save (±{args.half_width / 1e3:.0f}k)")
    cols = [c for c in CSV_COLUMNS if c in run.columns]
    header = ["save", "index", "D"] + cols + (["α_max", "α_argmax"] if alphas else [])
    print(" ".join(f"{h:>8}" for h in header))
    for k in probed:
        index = k * save_every
        st = at_index(run, cols + alphas, index, args.half_width)
        if st is None:
            print(f"{'save_' + str(k):>8} {index / 1e3:>7.0f}k  (hors CSV)")
            continue
        d_at = next((r["D"] for r in reversed(run.rows) if r["index"] <= index), None)
        row = [fmt(f"save_{k}"), fmt(f"{index / 1e3:.0f}k"), fmt(d_at)] \
            + [fmt(st.get(c)) for c in cols]
        if alphas:
            best = max(alphas, key=lambda a: -math.inf if math.isnan(st[a]) else st[a])
            row += [fmt(st[best]), fmt(best.replace("alpha_", ""))]
        print(" ".join(row))

    missing = [k for k in existing if k not in probed and k > 0]
    if missing:
        print(f"\nsaves non sondées : {missing}")


if __name__ == "__main__":
    main()
