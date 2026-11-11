import csv
import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
)
OUTPUTS = REPO / "outputs"

KEY_COLUMNS = [
    "r", "r_hit", "r_rcv", "r_aim", "hit", "kill", "fire", "s", "miss",
    "ev", "v", "Hc", "Hd", "kl", "clip", "skip",
]
SOURCE_DIRS = [
    "arenai_agent/src", "arenai_model/src", "arenai_core/src",
    "arenai_model/include", "arenai_core/include",
]
RUNNING_THRESHOLD_S = 15 * 60
UNCOMMITTED_WINDOW_S = 90 * 60


@dataclass
class Run:
    path: Path
    number: int
    config: dict
    columns: list
    rows: list
    launch: datetime
    last_write: datetime
    context: dict = field(default_factory=dict)

    @property
    def name(self):
        return self.path.name

    def col(self, name):
        return [r[name] for r in self.rows if name in r and not math.isnan(r[name])]


def resolve_run(arg):
    p = Path(arg)
    if p.is_dir() and (p / "metrics.csv").exists():
        return p.resolve()
    digits = "".join(c for c in Path(arg).name if c.isdigit())
    matches = sorted(OUTPUTS.glob(f"train_{int(digits)}_*")) if digits else []
    if not matches:
        raise SystemExit(f"run introuvable : {arg}")
    return matches[0].resolve()


def run_number(path):
    try:
        return int(path.name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _to_float(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


def load_run(arg):
    path = resolve_run(arg)
    with open(path / "metrics.csv", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = [h for h in next(reader) if h]
        rows = []
        for line in reader:
            values = [v for v in line if v != ""]
            if len(values) != len(header):
                continue
            row = dict(zip(header, map(_to_float, values)))
            if row.get("index", 0) > 0:
                rows.append(row)
    config_path = path / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    launch_src = config_path if config_path.exists() else path / "metrics.csv"
    return Run(
        path=path,
        number=run_number(path),
        config=config,
        columns=header,
        rows=rows,
        launch=datetime.fromtimestamp(launch_src.stat().st_mtime),
        last_write=datetime.fromtimestamp((path / "metrics.csv").stat().st_mtime),
    )


def previous_run(run):
    candidates = [
        p for p in OUTPUTS.glob("train_*")
        if (p / "config.json").exists() and (p / "metrics.csv").exists()
        and 0 <= run_number(p) < run.number
    ]
    return max(candidates, key=run_number) if candidates else None


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def config_value(run, key):
    for section in ("agent", "environment", "train"):
        if key in run.config.get(section, {}):
            return run.config[section][key]
    return None


def git(*args):
    res = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return res.stdout.strip()


def commit_before(ts):
    out = git("log", "-1", f"--before={ts.isoformat()}", "--format=%h|%ci|%s", "HEAD")
    if not out:
        return None
    h, date, subject = out.split("|", 2)
    return {"hash": h, "date": date, "subject": subject}


def commits_between(start, end):
    out = git(
        "log", f"--since={start.isoformat()}", f"--until={end.isoformat()}",
        "--reverse", "--format=%h|%ci|%s", "HEAD",
    )
    commits = []
    for line in out.splitlines():
        h, date, subject = line.split("|", 2)
        commits.append({"hash": h, "date": date, "subject": subject})
    return commits


def next_launch(run):
    later = [
        datetime.fromtimestamp((p / "config.json").stat().st_mtime)
        for p in OUTPUTS.glob("train_*")
        if (p / "config.json").exists() and run_number(p) > run.number
    ]
    later = [t for t in later if t > run.launch]
    return min(later) if later else None


def build_context(run):
    now = datetime.now()
    at_launch = commit_before(run.launch)
    after = commits_between(run.launch, run.last_write)
    nxt = next_launch(run)
    for c in after:
        date = datetime.fromisoformat(c["date"]).replace(tzinfo=None)
        delta = (date - run.launch).total_seconds()
        c["minutes_after_launch"] = round(delta / 60)
        c["maybe_run_code"] = delta <= UNCOMMITTED_WINDOW_S and (nxt is None or date < nxt)
    candidates = [c for c in after if c["maybe_run_code"]]
    hours = max((run.last_write - run.launch).total_seconds() / 3600, 1e-6)
    last_index = run.rows[-1]["index"] if run.rows else 0
    nb_tanks = config_value(run, "nb_tanks") or 1
    run.context = {
        "launch": run.launch.isoformat(timespec="minutes"),
        "last_write": run.last_write.isoformat(timespec="minutes"),
        "running": (now - run.last_write).total_seconds() < RUNNING_THRESHOLD_S,
        "hours": round(hours, 2),
        "steps_per_hour": round(last_index / hours),
        "tank_steps_per_hour": round(last_index * nb_tanks / hours),
        "commit_at_launch": at_launch,
        "effective_commit": candidates[-1] if candidates else at_launch,
        "next_run_launch": nxt.isoformat(timespec="minutes") if nxt else None,
        "commits_during_run": after,
    }
    return run.context


def code_diff(run_a, run_b):
    ca = (run_a.context or build_context(run_a))["effective_commit"]
    cb = (run_b.context or build_context(run_b))["effective_commit"]
    if not ca or not cb:
        return {}
    end = cb["hash"]
    log = git("log", "--reverse", "--format=%h %ci %s", f"{ca['hash']}..{end}")
    return {"from": ca["hash"], "to": end, "log": log.splitlines(),
            **substantive_diff(ca["hash"], end)}


def _is_trivial(line):
    body = line[1:].strip()
    return not body or body.startswith("//")


def substantive_diff(a, b, paths=None):
    out = git("diff", "-U0", "--no-color", a, b, "--", *(paths or SOURCE_DIRS))
    files, current = {}, None
    for line in out.splitlines():
        if line.startswith("diff --git"):
            current = line.split(" b/", 1)[1]
            files[current] = [0, 0]
        elif current is None or line.startswith(("+++", "---")):
            continue
        elif line.startswith("+") and not _is_trivial(line):
            files[current][0] += 1
        elif line.startswith("-") and not _is_trivial(line):
            files[current][1] += 1
    changed = {f: c for f, c in files.items() if c[0] or c[1]}
    width = max((len(f) for f in changed), default=0)
    stat = "\n".join(f"{f.ljust(width)}  +{c[0]} -{c[1]}" for f, c in changed.items())
    return {"stat": stat, "comment_only_files": len(files) - len(changed)}


def config_diff(run_a, run_b):
    fa, fb = flatten(run_a.config), flatten(run_b.config)
    skip = {"train.output_folder"}
    diff = {}
    for k in sorted(set(fa) | set(fb)):
        if k in skip:
            continue
        if fa.get(k) != fb.get(k):
            diff[k] = (fa.get(k), fb.get(k))
    return diff


def mean(values):
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def d_milestones(run):
    out = []
    seen = set()
    for r in run.rows:
        d = r.get("D")
        if d is None or math.isnan(d) or d in seen:
            continue
        seen.add(d)
        out.append((d, r["index"]))
    return out


def per_d_level(run, columns):
    levels = {}
    for r in run.rows:
        d = r.get("D")
        if d is None or math.isnan(d):
            continue
        levels.setdefault(d, []).append(r)
    out = {}
    for d, rows in sorted(levels.items()):
        stats = {"n": len(rows), "from": rows[0]["index"], "to": rows[-1]["index"]}
        for c in columns:
            stats[c] = mean([x[c] for x in rows if c in x])
        out[d] = stats
    return out


def windows(run, columns, nb=10):
    if not run.rows:
        return []
    last = run.rows[-1]["index"]
    width = last / nb
    out = []
    for i in range(nb):
        lo, hi = i * width, (i + 1) * width
        rows = [r for r in run.rows if lo < r["index"] <= hi]
        if not rows:
            continue
        stats = {"from": lo, "to": hi, "n": len(rows), "D": rows[-1].get("D", float("nan"))}
        for c in columns:
            stats[c] = mean([x[c] for x in rows if c in x])
        out.append(stats)
    return out


def at_index(run, columns, index, half_width):
    rows = [r for r in run.rows if index - half_width < r["index"] <= index + half_width]
    if not rows:
        return None
    stats = {"n": len(rows), "D": rows[-1].get("D", float("nan"))}
    for c in columns:
        stats[c] = mean([x[c] for x in rows if c in x])
    return stats


def extremes(run, columns):
    out = {}
    for c in columns:
        pts = [(r[c], r["index"], r.get("D")) for r in run.rows if c in r and not math.isnan(r[c])]
        if not pts:
            continue
        hi, lo = max(pts), min(pts)
        out[c] = {"max": hi, "min": lo, "last": pts[-1]}
    return out


def alpha_columns(run):
    return [c for c in run.columns if c.startswith("alpha_")]


def signals(run):
    out = []
    n = len(run.rows)
    if n < 5:
        return ["moins de 5 lignes : rien d'interprétable"]
    tail = run.rows[int(n * 0.8):]
    target_kl = config_value(run, "target_kl")
    initial_alpha = config_value(run, "initial_alpha")

    def tail_mean(c):
        return mean([r[c] for r in tail if c in r])

    if "ev" in run.columns:
        ev = tail_mean("ev")
        if ev < 0.2:
            out.append(f"ev moyen {ev:.2f} sur les 20 % finaux : critic faible")
        neg = sum(1 for r in run.rows if r.get("ev", 0) < 0) / n
        if neg > 0.2:
            out.append(f"ev < 0 sur {neg:.0%} des lignes")
    if "skip" in run.columns and tail_mean("skip") > 0.1:
        out.append(f"skip {tail_mean('skip'):.2f} en fin de run : régime KL")
    if "kl" in run.columns and target_kl and tail_mean("kl") > target_kl:
        out.append(f"kl {tail_mean('kl'):.3f} > target_kl {target_kl:.3f}")
    if "Hc" in run.columns and tail_mean("Hc") < 0:
        out.append(f"Hc {tail_mean('Hc'):.2f} < 0 en fin de run")
    if initial_alpha:
        for c in alpha_columns(run):
            peak = max(r[c] for r in run.rows if c in r)
            if peak > 1.5 * initial_alpha:
                out.append(f"{c} a quitté le plancher (max {peak:.4f} vs init {initial_alpha:.4f})")
    if "fire" in run.columns:
        peak = max(r["fire"] for r in run.rows[1:]) if n > 1 else 0
        if tail_mean("fire") < 0.05 and peak > 0.2:
            out.append(f"fire {tail_mean('fire'):.3f}/s en fin (pic {peak:.2f}) : gâchette éteinte")
    for c in ("hit", "kill"):
        if c not in run.columns:
            continue
        peak = max(r[c] for r in run.rows)
        if peak > 0 and tail_mean(c) < 0.7 * peak:
            idx = max(run.rows, key=lambda r: r[c])["index"]
            out.append(f"{c} final {tail_mean(c):.3f} < 70 % du pic {peak:.3f} (à {idx / 1e3:.0f} k)")
    if "D" in run.columns:
        last_d = run.rows[-1]["D"]
        first_at_last = next(r["index"] for r in run.rows if r["D"] == last_d)
        span = run.rows[-1]["index"]
        if last_d < (config_value(run, "final_spawn_width") or float("inf")) and \
                span - first_at_last > 0.3 * span:
            out.append(f"D bloqué à {last_d:.0f} depuis {first_at_last / 1e3:.0f} k")
    return out


def summary(run):
    cols = [c for c in KEY_COLUMNS if c in run.columns]
    alphas = alpha_columns(run)
    return {
        "run": run.name,
        "path": str(run.path),
        "columns": run.columns,
        "nb_rows": len(run.rows),
        "last_index": run.rows[-1]["index"] if run.rows else 0,
        "context": run.context or build_context(run),
        "config": run.config,
        "d_milestones": d_milestones(run),
        "per_d_level": per_d_level(run, cols + alphas),
        "windows": windows(run, cols + alphas),
        "extremes": extremes(run, cols + alphas),
        "signals": signals(run),
    }


def fmt_value(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def fmt(v, width=8):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-".rjust(width)
    if isinstance(v, float):
        a = abs(v)
        if a == 0:
            s = "0"
        elif a >= 10 and v.is_integer():
            s = f"{v:.0f}"
        elif a >= 1000:
            s = f"{v:.0f}"
        elif a >= 10:
            s = f"{v:.1f}"
        elif a >= 0.01:
            s = f"{v:.3f}"
        else:
            s = f"{v:.1e}"
        return s.rjust(width)
    return str(v).rjust(width)


def table(header, rows, width=8):
    first = max([len(str(header[0]))] + [len(fmt(r[0]).strip()) for r in rows])
    lines = [str(header[0]).ljust(first) + " " + " ".join(str(h).rjust(width) for h in header[1:])]
    for r in rows:
        lines.append(fmt(r[0]).strip().ljust(first) + " " + " ".join(fmt(v, width) for v in r[1:]))
    return "\n".join(lines)
