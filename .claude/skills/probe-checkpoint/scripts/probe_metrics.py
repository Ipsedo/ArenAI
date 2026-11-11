import csv
import math
import statistics as st
from collections import defaultdict

ALIGNED_DEG = 5.0
OFF_AXIS_DEG = 20.0
CLOSE_M = 40.0
LATE_STEP = 10
SIGMA_CEILING = 0.69
MU_SATURATED = 1.5
ANGLE_BINS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 45), (45, 180)]
SHOT_BINS = [(0, 5), (5, 20), (20, 180)]


def load(path):
    with open(path) as f:
        rows = []
        for r in csv.DictReader(f, delimiter=";"):
            try:
                rows.append({k: float(v) for k, v in r.items()})
            except (TypeError, ValueError):
                continue
        return rows


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return st.mean(xs) if xs else float("nan")


def quantile(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))] if xs else float("nan")


def ratio(a, b):
    return a / b if b and not math.isnan(a) and not math.isnan(b) else float("nan")


def tanh_normal_entropy(mu, sigma, n=32):
    h = 0.5 * math.log(2 * math.pi * math.e * sigma * sigma)
    acc = 0.0
    for k in range(n):
        x = mu + sigma * st.NormalDist().inv_cdf((k + 0.5) / n)
        t = math.tanh(x)
        acc += math.log(max(1 - t * t, 1e-12))
    return h + acc / n


def analyze(decisions_path, step_seconds):
    rows = load(decisions_path)
    shots_path = decisions_path.replace(".csv", "_shots.csv")
    try:
        shots = load(shots_path)
    except FileNotFoundError:
        shots = []

    late = [r for r in rows if r["step"] >= LATE_STEP]
    armed = [r for r in late if r["can_fire"] > 0.5 and r["shells"] > 0]
    out = {
        "decisions": len(rows),
        "episodes": int(max((r["episode"] for r in rows), default=-1)) + 1,
        "tanks": int(max((r["tank"] for r in rows), default=-1)) + 1,
        "step_seconds": step_seconds,
        "armed_fraction": ratio(len(armed), len(late)),
    }

    # trigger
    def p_fire(sel):
        return mean(r["p_fire"] for r in sel)

    by_bin = {}
    for lo, hi in ANGLE_BINS:
        sub = [r for r in armed if lo <= r["best_angle"] < hi]
        by_bin[f"{lo}-{hi}"] = {
            "n": len(sub), "p_fire": p_fire(sub), "fire": mean(r["fire"] for r in sub),
            "p_zoom": mean(r["p_zoom"] for r in sub),
        }
    aligned = [r for r in armed if r["best_angle"] < ALIGNED_DEG]
    aligned_close = [r for r in aligned if 0 <= r["best_distance"] < CLOSE_M]
    off = [r for r in armed if r["best_angle"] >= OFF_AXIS_DEG]
    pa, pac, po = p_fire(aligned), p_fire(aligned_close), p_fire(off)
    out["trigger"] = {
        "p_fire_armed": p_fire(armed),
        "p_fire_aligned": pa,
        "p_fire_aligned_close": pac,
        "p_fire_off_axis": po,
        "ratio": ratio(pac, po),
        "ratio_angle_only": ratio(pa, po),
        "gap": pac - po,
        "n_aligned": len(aligned),
        "n_aligned_close": len(aligned_close),
        "n_off_axis": len(off),
        "deterministic_fires_aligned": mean(r["p_fire"] > 0.5 for r in aligned),
        "deterministic_fires_off_axis": mean(r["p_fire"] > 0.5 for r in off),
        "by_angle": by_bin,
    }
    if len(armed) > 2:
        try:
            out["trigger"]["corr_p_fire_p_zoom"] = st.correlation(
                [r["p_fire"] for r in armed], [r["p_zoom"] for r in armed])
        except st.StatisticsError:
            out["trigger"]["corr_p_fire_p_zoom"] = float("nan")

    # turret
    out["turret"] = {
        f"below_{t}deg": mean(r["best_angle"] < t for r in late) for t in (5, 10, 20)
    }
    out["turret"]["median_best_angle"] = quantile([r["best_angle"] for r in late], 0.5)
    out["turret"]["median_best_distance"] = quantile(
        [r["best_distance"] for r in late if r["best_distance"] >= 0], 0.5)
    out["turret"]["median_nearest_distance"] = quantile([r["distance"] for r in late], 0.5)

    # policy heads
    heads = {}
    for d in ("canon_x", "canon_y"):
        s = [r[f"sigma_{d}"] for r in late]
        heads[f"sigma_{d}"] = {
            "p10": quantile(s, 0.1), "p50": quantile(s, 0.5), "p90": quantile(s, 0.9),
            "at_ceiling": mean(x >= SIGMA_CEILING for x in s),
        }
        sample = late[::max(1, len(late) // 3000)]
        heads[f"H_{d}"] = mean(tanh_normal_entropy(r[f"mu_{d}"], r[f"sigma_{d}"]) for r in sample)
    for d in ("dir_x", "dir_y", "canon_x", "canon_y"):
        m = [r[f"mu_{d}"] for r in late]
        heads[f"mu_{d}"] = {
            "mean": mean(m), "abs_p50": quantile([abs(x) for x in m], 0.5),
            "saturated": mean(abs(x) > MU_SATURATED for x in m),
        }
    heads["p_zoom"] = mean(r["p_zoom"] for r in late)
    out["heads"] = heads

    # reserve
    per = defaultdict(list)
    for r in rows:
        per[(r["episode"], r["tank"])].append(r)
    first_empty, lengths, disarmed = [], [], []
    for seq in per.values():
        seq.sort(key=lambda r: r["step"])
        lengths.append(len(seq) * step_seconds)
        empty = [r["step"] for r in seq if r["shells"] <= 0]
        if empty:
            first_empty.append(empty[0] * step_seconds)
        disarmed.append(mean(r["shells"] <= 0 for r in seq))
    out["reserve"] = {
        "tank_episodes": len(per),
        "emptied": ratio(len(first_empty), len(per)),
        "first_empty_s_p50": quantile(first_empty, 0.5),
        "first_empty_s_p10": quantile(first_empty, 0.1),
        "first_empty_s_p90": quantile(first_empty, 0.9),
        "disarmed_fraction": mean(disarmed),
        "life_s_p50": quantile(lengths, 0.5),
    }

    # shots
    fired = sum(r["fire"] > 0.5 and r["can_fire"] > 0.5 for r in rows)
    out["shots"] = {"fired": fired, "per_tank_second": ratio(fired, len(rows) * step_seconds)}
    if shots:
        n = sum(s["group_size"] for s in shots)
        h = sum(s["group_hits"] for s in shots)
        out["shots"].update({
            "landed": n, "hit_per_shell": ratio(h, n),
            "miss_m_p50": quantile([s["group_miss"] for s in shots], 0.5),
        })
        bins = {}
        for lo, hi in SHOT_BINS:
            sub = [s for s in shots if lo <= s["fire_best_angle"] < hi]
            n2 = sum(s["group_size"] for s in sub)
            bins[f"{lo}-{hi}"] = {
                "share": ratio(n2, n), "hit_per_shell": ratio(sum(s["group_hits"] for s in sub), n2)}
        out["shots"]["by_angle"] = bins
        aligned_close_shots = sum(
            s["group_size"] for s in shots
            if s["fire_best_angle"] < ALIGNED_DEG and 0 <= s["fire_best_distance"] < CLOSE_M)
        out["shots"]["aligned_close_share"] = ratio(aligned_close_shots, n)
        out["shots"]["enrichment"] = ratio(
            bins["0-5"]["share"], mean(r["best_angle"] < ALIGNED_DEG for r in armed))
    return out


HEADLINE = [
    ("ratio", lambda m: m["trigger"]["ratio"]),
    ("p_al", lambda m: m["trigger"]["p_fire_aligned_close"]),
    ("p_off", lambda m: m["trigger"]["p_fire_off_axis"]),
    ("p_arm", lambda m: m["trigger"]["p_fire_armed"]),
    ("hit/ob", lambda m: m["shots"].get("hit_per_shell", float("nan"))),
    ("h<5°", lambda m: m["shots"].get("by_angle", {}).get("0-5", {}).get("hit_per_shell", float("nan"))),
    ("tirs<5°", lambda m: m["shots"].get("by_angle", {}).get("0-5", {}).get("share", float("nan"))),
    ("enrich", lambda m: m["shots"].get("enrichment", float("nan"))),
    ("vide_s", lambda m: m["reserve"]["first_empty_s_p50"]),
    ("%vident", lambda m: m["reserve"]["emptied"]),
    ("%désarmé", lambda m: m["reserve"]["disarmed_fraction"]),
    ("tir/s", lambda m: m["shots"]["per_tank_second"]),
    ("tour<5°", lambda m: m["turret"]["below_5deg"]),
    ("σx", lambda m: m["heads"]["sigma_canon_x"]["p50"]),
    ("σy", lambda m: m["heads"]["sigma_canon_y"]["p50"]),
    ("σ@plaf", lambda m: max(m["heads"]["sigma_canon_x"]["at_ceiling"],
                             m["heads"]["sigma_canon_y"]["at_ceiling"])),
    ("|μ|cx", lambda m: m["heads"]["mu_canon_x"]["abs_p50"]),
    ("μsat", lambda m: max(m["heads"][f"mu_{d}"]["saturated"]
                           for d in ("dir_x", "dir_y", "canon_x", "canon_y"))),
    ("p_zoom", lambda m: m["heads"]["p_zoom"]),
]
