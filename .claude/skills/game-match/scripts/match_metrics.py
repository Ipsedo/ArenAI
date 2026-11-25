#!/usr/bin/env python3
"""Game outcomes of arenai_agent_match exports. Stdlib only.

usage: match_metrics.py <name>=<prefix> [<name>=<prefix> ...] [--json out.json]
<prefix> is the --output of arenai_agent_match (files <prefix>_{tanks,player,matches}.csv).
Each metric is a per-match value; printed as mean ± sd over the matches.
"""
import csv, json, statistics, sys
from collections import defaultdict

DRY_TIMES = (10, 30, 60, 120)


def rows(path):
    with open(path) as f:
        yield from csv.DictReader(f, delimiter=";")


def median(values):
    return statistics.median(values) if values else float("nan")


def one_match(tanks, player, info):
    by_tank = defaultdict(list)
    for r in tanks:
        by_tank[r["tank"]].append(r)
    n = len(by_tank)
    seconds = float(info["seconds"])
    step = float(tanks[0]["second"])

    m = {"durée_s": seconds, "fin_à_sec": float(info["end"] == "all_dry")}

    # 1. trouve les ennemis: canon axis within 5° of a living target, while armed
    armed = [r for r in tanks if r["alive"] == "1" and r["out_of_shells"] == "0"]
    m["trouve: armé à <5° d'une cible"] = (
        sum(float(r["best_angle"]) < 5 for r in armed) / len(armed) if armed else float("nan"))
    first = []
    for rs in by_tank.values():
        t = next((float(r["second"]) for r in rs
                  if r["alive"] == "1" and float(r["best_angle"]) < 5 and float(r["second"]) > 1), None)
        first.append(t)
    m["trouve: 1re cible <5° (s, médiane)"] = median([t for t in first if t is not None])

    # 2. reste dangereux: living tanks with an empty reserve
    for t in DRY_TIMES:
        at = [r for r in tanks if abs(float(r["second"]) - t) < step / 2 and r["alive"] == "1"]
        m[f"dangereux: vivants à sec à {t} s"] = (
            sum(r["out_of_shells"] == "1" for r in at) / len(at) if at else float("nan"))
    alive = [r for r in tanks if r["alive"] == "1"]
    m["dangereux: temps vivant à sec"] = (
        sum(r["out_of_shells"] == "1" for r in alive) / len(alive) if alive else float("nan"))
    m["dangereux: armés en fin / tanks"] = float(info["enemies_armed"]) / n

    # 3. touche le joueur (idle player tank at its game spawn)
    impacts = [(float(r["second"]), int(r["impacts"])) for r in player if int(r["impacts"]) > 0]
    death = next((float(r["second"]) for r in player if r["alive"] == "0"), None)
    m["joueur: impacts reçus"] = sum(i for _, i in impacts)
    m["joueur: 1er impact (s)"] = impacts[0][0] if impacts else float("nan")
    m["joueur: tué"] = float(death is not None)
    m["joueur: mort à (s)"] = death if death is not None else float("nan")

    # 4. touche et tue les autres tanks (player included in hits)
    fired = sum(int(r["fired"]) for r in tanks)
    hits = sum(int(r["hits"]) for r in tanks)
    m["combat: touches / tank"] = hits / n
    m["combat: kills / tank"] = sum(int(r["kills"]) for r in tanks) / n
    m["combat: obus tirés / tank"] = fired / n
    m["combat: touche / obus"] = hits / fired if fired else float("nan")
    m["combat: tanks sans touche"] = sum(
        all(int(r["hits"]) == 0 for r in rs) for rs in by_tank.values()) / n

    # 5. tenue visible: canon elevation, armed vs dry
    dry = [float(r["canon_elevation"]) for r in alive if r["out_of_shells"] == "1"]
    m["tenue: élévation canon armé (°, médiane)"] = median(
        [float(r["canon_elevation"]) for r in armed])
    m["tenue: élévation canon à sec (°, médiane)"] = median(dry)
    m["tenue: armé canon < -5°"] = (
        sum(float(r["canon_elevation"]) < -5 for r in armed) / len(armed) if armed else float("nan"))
    return m


def checkpoint(prefix):
    tanks, player = defaultdict(list), defaultdict(list)
    for r in rows(prefix + "_tanks.csv"):
        tanks[r["match"]].append(r)
    for r in rows(prefix + "_player.csv"):
        player[r["match"]].append(r)
    return [one_match(tanks[i["match"]], player[i["match"]], i)
            for i in rows(prefix + "_matches.csv")]


def main():
    args = [a for a in sys.argv[1:] if "=" in a]
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    results = {name: checkpoint(prefix) for name, prefix in (a.split("=", 1) for a in args)}
    names = list(results)
    summary = {}
    print(f"{'':44}" + "".join(f"{n:>22}" for n in names))
    print(f"{'matchs':44}" + "".join(f"{len(results[n]):>22}" for n in names))
    for key in results[names[0]][0]:
        line = f"{key:44}"
        for n in names:
            v = [m[key] for m in results[n] if m[key] == m[key]]
            mean = statistics.fmean(v) if v else float("nan")
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            summary.setdefault(n, {})[key] = {"mean": mean, "sd": sd, "n": len(v)}
            line += f"{mean:>12.2f} ± {sd:<5.2f}({len(v)})"
        print(line)
    if out:
        with open(out, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
