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
    parser = argparse.ArgumentParser(description="Numeric summary of an ArenAI run")
    parser.add_argument("run", help="run folder, or its number (504)")
    parser.add_argument("--prev", help="reference run for diffs (default: previous number)")
    parser.add_argument("--json", action="store_true", help="writes <run>/analysis_summary.json")
    args = parser.parse_args()

    run = load_run(args.run)
    ctx = build_context(run)
    s = summary(run)
    cols = [c for c in KEY_COLUMNS if c in run.columns]
    alphas = alpha_columns(run)

    print(f"# {run.name}")
    print(f"launched {ctx['launch']} (config.json mtime), last CSV write {ctx['last_write']}"
          f"{' — EN COURS' if ctx['running'] else ''}")
    print(f"{s['nb_rows']} rows, final index {k(s['last_index'])}, {ctx['hours']} h, "
          f"{ctx['steps_per_hour']} steps/h, {ctx['tank_steps_per_hour']} tank-steps/h")

    print("\n## Code")
    c = ctx["commit_at_launch"]
    print(f"commit at launch: {c['hash']} {c['date']} {c['subject']}" if c else "no commit")
    e = ctx["effective_commit"]
    if e and c and e["hash"] != c["hash"]:
        print(f"estimated effective commit: {e['hash']} (confirm against config.json / memory)")
    for cm in ctx["commits_during_run"]:
        tag = "  <- may contain the run's code (launched uncommitted)" if cm["maybe_run_code"] else ""
        print(f"  during the run +{cm['minutes_after_launch']} min: {cm['hash']} {cm['subject']}{tag}")

    prev_path = args.prev or previous_run(run)
    if not prev_path:
        print("\n## Differences: no previous run in outputs/")
    else:
        prev = load_run(prev_path)
        build_context(prev)
        print(f"\n## Differences vs {prev.name}")
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
                  or "  no change outside comments")
            if cd["comment_only_files"]:
                print(f"  (+ {cd['comment_only_files']} files changed only in comments/whitespace)")
        added = [c for c in run.columns if c not in prev.columns]
        removed = [c for c in prev.columns if c not in run.columns]
        if added or removed:
            print(f"colonnes CSV : +{added} -{removed}")

    print("\n## Config (agent / environment)")
    for section in ("agent", "environment"):
        items = run.config.get(section, {})
        print(f"{section}: " + ", ".join(f"{key}={fmt_value(v)}" for key, v in items.items()))

    print("\n## D milestones (index of first appearance)")
    print("  " + ", ".join(f"D {d:.0f} @ {k(i)}" for d, i in s["d_milestones"]))

    print("\n## Means per D level")
    show = ["n", "from", "to"] + cols
    print(table(["D"] + show, [[d] + [st[c] for c in show] for d, st in s["per_d_level"].items()]))

    print("\n## Windows (10 index slices)")
    show = ["to", "D"] + cols
    print(table(show, [[w[c] for c in show] for w in s["windows"]]))
    if alphas:
        print()
        show = ["to"] + alphas
        print(table(["to"] + [a.replace("alpha_", "α_") for a in alphas],
                    [[w[c] for c in show] for w in s["windows"]], width=10))

    print("\n## Extremes (value @ index, D)")
    for c, e in s["extremes"].items():
        mx, mn, last = e["max"], e["min"], e["last"]
        print(f"  {c:>14}: max {fmt(mx[0]).strip()} @ {k(mx[1])} D{fmt(mx[2]).strip()}"
              f" | min {fmt(mn[0]).strip()} @ {k(mn[1])} | dernier {fmt(last[0]).strip()}")

    print("\n## Automatic signals (to verify, not verdicts)")
    for sig in s["signals"] or ["none"]:
        print(f"  - {sig}")

    if args.json:
        out = run.path / "analysis_summary.json"
        out.write_text(json.dumps(s, indent=1, default=str))
        print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
