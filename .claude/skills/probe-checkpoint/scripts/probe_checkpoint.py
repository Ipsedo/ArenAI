import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "analyze-run" / "scripts"))

from arenai_run import (  # noqa: E402
    REPO, build_context, commit_before, config_value, fmt, git, load_run, substantive_diff,
)
from probe_metrics import HEADLINE, analyze  # noqa: E402

BINARIES = [
    REPO / "cmake-build-release" / "arenai_agent" / "arenai_agent_probe",
    REPO / "build" / "arenai_agent" / "arenai_agent_probe",
]
INFERENCE_PATHS = [
    "arenai_agent/probe",
    "arenai_agent/src/networks",
    "arenai_agent/src/distributions",
    "arenai_agent/src/agents/torch_types.cpp",
    "arenai_agent/src/agents/torch_types.h",
    "arenai_agent/src/agents/ppo_liquid/liquid_hidden_state.cpp",
    "arenai_agent/src/core/train_environment.cpp",
    "arenai_model/src",
    "arenai_model/include",
    "arenai_core/src",
    "arenai_core/include",
]


def resolve_save(arg, save):
    p = Path(arg)
    if p.is_dir() and p.name.startswith("save_") and save is None:
        return load_run(p.parent), p.name
    run = load_run(arg)
    if save is None:
        raise SystemExit("préciser --save save_N quand l'entrée est un run")
    return run, save if str(save).startswith("save_") else f"save_{save}"


def repr_signature(save_dir):
    path = save_dir / "actor_repr.txt"
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    return [re.sub(r"\(.*\)", "", line).strip() for line in lines
            if re.search(r"^\s+(mu|sigma|\d+): ", line) and "Conv2d" not in line
            and "GroupNorm" not in line]


def compatibility(run, binary):
    built = datetime.fromtimestamp(binary.stat().st_mtime)
    binary_commit = commit_before(built)
    run_commit = run.context["effective_commit"]
    diff = substantive_diff(run_commit["hash"], binary_commit["hash"], INFERENCE_PATHS) \
        if run_commit and binary_commit else {"stat": "", "comment_only_files": 0}
    dirty = git("status", "--porcelain", "--", *INFERENCE_PATHS)
    dirty_before_build = []
    for line in dirty.splitlines():
        f = REPO / line[3:]
        if f.exists() and datetime.fromtimestamp(f.stat().st_mtime) < built:
            dirty_before_build.append(line[3:])
    return {
        "binary": str(binary),
        "built": built.isoformat(timespec="minutes"),
        "binary_commit": binary_commit,
        "run_commit": run_commit,
        "inference_diff": diff["stat"],
        "uncommitted_in_binary": dirty_before_build,
    }


def run_probe(run, save, binary, side, args, out_csv):
    env = run.config.get("environment", {})
    cmd = [
        str(binary),
        "--run_folder", str(run.path),
        "--save", save,
        "--resources_folder", run.config.get("train", {}).get(
            "resources_folder", str(REPO / "resources")),
        "--output", str(out_csv),
        "--nb_episodes", str(args.episodes),
        "--spawn_side", str(side),
    ]
    for key in ("shells", "shells_per_hit", "max_shells"):
        if key in env:
            cmd += [f"--{key}", str(env[key])]
    if args.cuda:
        cmd.append("--cuda")
    if args.deterministic:
        cmd.append("--deterministic")
    log = out_csv.with_suffix(".log")
    print(f"$ {' '.join(cmd)}", flush=True)
    with open(log, "w") as f:
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if res.returncode != 0:
        raise SystemExit(f"sonde en échec (code {res.returncode}), voir {log}")


def print_detail(side, m):
    t, tu, h, r, s = m["trigger"], m["turret"], m["heads"], m["reserve"], m["shots"]
    print(f"\n### côté {side:g} — {m['decisions']} décisions, {m['episodes']} épisodes, "
          f"{m['tanks']} tanks, armé {fmt(m['armed_fraction']).strip()}")
    print("p_fire armé (step ≥ 10) par best_angle :")
    for b, v in t["by_angle"].items():
        print(f"  [{b:>6}°) n={v['n']:6d} p_fire {fmt(v['p_fire']).strip():>6}"
              f" p_zoom {fmt(v['p_zoom']).strip():>6}")
    print(f"  aligné<5°&<40m {fmt(t['p_fire_aligned_close']).strip()} (n {t['n_aligned_close']}),"
          f" hors axe≥20° {fmt(t['p_fire_off_axis']).strip()} (n {t['n_off_axis']}),"
          f" ratio {fmt(t['ratio']).strip()}, écart {fmt(t['gap']).strip()} ; aligné<5° seul "
          f"{fmt(t['p_fire_aligned']).strip()} (n {t['n_aligned']}), ratio "
          f"{fmt(t['ratio_angle_only']).strip()}")
    print(f"  déterministe (p>0,5) : aligné {fmt(t['deterministic_fires_aligned']).strip()}, "
          f"hors axe {fmt(t['deterministic_fires_off_axis']).strip()} ; corr(p_fire, p_zoom) "
          f"{fmt(t.get('corr_p_fire_p_zoom', float('nan'))).strip()}")
    print(f"tourelle : <5° {fmt(tu['below_5deg']).strip()}, <10° {fmt(tu['below_10deg']).strip()},"
          f" <20° {fmt(tu['below_20deg']).strip()}, angle médian {fmt(tu['median_best_angle']).strip()}°,"
          f" distance cible médiane {fmt(tu['median_best_distance']).strip()} m")
    for d in h.get("aim_dims", ("canon_x", "canon_y")):
        sg = h[f"sigma_{d}"]
        print(f"σ_{d} p10/p50/p90 {fmt(sg['p10']).strip()}/{fmt(sg['p50']).strip()}/"
              f"{fmt(sg['p90']).strip()}, au plafond {fmt(sg['at_ceiling']).strip()}, "
              f"H estimée {fmt(h[f'H_{d}']).strip()}")
    print("μ : " + ", ".join(
        f"{d} moy {fmt(h[f'mu_{d}']['mean']).strip()} |μ|p50 {fmt(h[f'mu_{d}']['abs_p50']).strip()}"
        f" sat {fmt(h[f'mu_{d}']['saturated']).strip()}"
        for d in h.get("all_dims", ("dir_x", "dir_y", "canon_x", "canon_y"))))
    print(f"réserve : {fmt(r['emptied']).strip()} des tank-épisodes vident, 1re réserve vide "
          f"p10/p50/p90 {fmt(r['first_empty_s_p10']).strip()}/{fmt(r['first_empty_s_p50']).strip()}/"
          f"{fmt(r['first_empty_s_p90']).strip()} s, désarmé {fmt(r['disarmed_fraction']).strip()},"
          f" vie médiane {fmt(r['life_s_p50']).strip()} s")
    print(f"obus : {s['fired']} tirs ({fmt(s['per_tank_second']).strip()}/tank/s), "
          f"hit/obus {fmt(s.get('hit_per_shell', math.nan)).strip()}, ratage médian "
          f"{fmt(s.get('miss_m_p50', math.nan)).strip()} m, tirs alignés&proches "
          f"{fmt(s.get('aligned_close_share', math.nan)).strip()}, enrichissement "
          f"{fmt(s.get('enrichment', math.nan)).strip()}")
    for b, v in s.get("by_angle", {}).items():
        print(f"  tirés à [{b}°) : part {fmt(v['share']).strip()}, hit/obus "
              f"{fmt(v['hit_per_shell']).strip()}")


def main():
    parser = argparse.ArgumentParser(description="Sonde un checkpoint ArenAI et résume les métriques")
    parser.add_argument("checkpoint", help="outputs/train_NNN_*/save_K, ou un run avec --save")
    parser.add_argument("--save", help="save_K ou K quand l'entrée est un run")
    parser.add_argument("--sides", default="550,1000", help="côtés de spawn, séparés par des virgules")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--binary", help="binaire arenai_agent_probe (défaut : le plus récent)")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-cuda", dest="cuda", action="store_false")
    parser.add_argument("--check-only", action="store_true", help="compatibilité seulement")
    parser.add_argument("--analyze-only", action="store_true", help="réanalyse les CSV existants")
    parser.add_argument("--force", action="store_true", help="sonder malgré un diff d'inférence")
    args = parser.parse_args()

    run, save = resolve_save(args.checkpoint, args.save)
    build_context(run)
    save_dir = run.path / save
    if not (save_dir / "actor.pt").exists():
        raise SystemExit(f"{save_dir}/actor.pt introuvable")
    k = int(save.split("_")[1])
    save_every = config_value(run, "save_every") or 0
    index = k * save_every
    d_at = next((r["D"] for r in reversed(run.rows) if r["index"] <= index and "D" in r), None)

    binaries = [Path(args.binary)] if args.binary else [b for b in BINARIES if b.exists()]
    if not binaries:
        raise SystemExit("aucun binaire arenai_agent_probe : utiliser le skill build")
    binary = max(binaries, key=lambda b: b.stat().st_mtime)
    compat = compatibility(run, binary)

    print(f"# sonde {run.name}/{save} — index {index / 1e3:.0f}k, D {d_at}")
    print(f"binaire : {compat['binary']} (compilé {compat['built']}, "
          f"commit {compat['binary_commit']['hash']} {compat['binary_commit']['subject']})")
    rc = compat["run_commit"]
    print(f"code du run : {rc['hash']} {rc['subject']}" if rc else "code du run : inconnu")
    print("tête σ/μ du checkpoint : " + " | ".join(repr_signature(save_dir)[-7:]))
    if compat["uncommitted_in_binary"]:
        print("modifs non commitées antérieures au build (probablement dans le binaire) : "
              + ", ".join(compat["uncommitted_in_binary"]))
    if compat["inference_diff"]:
        print("DIFF D'INFÉRENCE run → binaire (réseau, distributions, observation, environnement) :")
        print("\n".join(f"  {line}" for line in compat["inference_diff"].splitlines()))
        if not (args.force or args.analyze_only or args.check_only):
            raise SystemExit("incompatibilité possible : vérifier le diff, puis --force si sans effet")
    else:
        print("aucun diff d'inférence entre le code du run et celui du binaire")
    if args.check_only:
        return

    out_dir = run.path / "probes"
    out_dir.mkdir(exist_ok=True)
    step_seconds = (config_value(run, "action_repeat") or 3) * \
        (config_value(run, "wanted_frequency") or 1 / 30)
    suffix = "_det" if args.deterministic else ""
    results = {}
    for side in [float(s) for s in args.sides.split(",")]:
        out_csv = out_dir / f"{save}_side{side:g}{suffix}.csv"
        if not args.analyze_only:
            run_probe(run, save, binary, side, args, out_csv)
        metrics = analyze(str(out_csv), step_seconds)
        meta = {
            "run": run.name, "save": save, "save_index": k, "index": index, "D_at_save": d_at,
            "side": side, "episodes": args.episodes, "deterministic": args.deterministic,
            "probed_at": datetime.now().isoformat(timespec="minutes"), **compat,
        }
        results[side] = metrics
        (out_csv.with_suffix(".json")).write_text(
            json.dumps({"meta": meta, "metrics": metrics}, indent=1, default=str))
        print_detail(side, metrics)

    print("\n## Résumé")
    header = ["côté"] + [h for h, _ in HEADLINE]
    print(" ".join(f"{h:>8}" for h in header))
    for side, m in results.items():
        print(" ".join([fmt(side)] + [fmt(f(m)) for _, f in HEADLINE]))
    print(f"\nrésultats : {out_dir}/{save}_side*{suffix}.json")


if __name__ == "__main__":
    main()
