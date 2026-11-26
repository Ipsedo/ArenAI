import argparse
import socket
from datetime import datetime

from arenai_run import (
    KNOWLEDGE_RUNS, REPO, SOURCE_DIRS, build_context, git, knowledge_dir, load_run,
)

UNKNOWN = "inconnu"


def describe(commit):
    return f"`{commit['hash']}` — {commit['subject']}" if commit else UNKNOWN


def launch_diff(run, context, folder):
    if not context["running"]:
        return f"{UNKNOWN} (identité relevée après la fin du run)"
    if not git("status", "--porcelain", "--", *SOURCE_DIRS):
        return "aucun (arbre propre, run en cours)"
    (folder / "launch.patch").write_text(git("diff", "HEAD", "--", *SOURCE_DIRS) + "\n")
    stamp = datetime.now().isoformat(sep=" ", timespec="seconds")
    return f"non commité — `launch.patch` capturé le {stamp}, run en cours"


def create(run, host, imported):
    context = build_context(run)
    stamp = run.launch.strftime("%Y-%m-%d-%H-%M-%S")
    folder = KNOWLEDGE_RUNS / f"{stamp}_{host}_{run.name}"
    folder.mkdir(parents=True)
    candidates = [c for c in context["commits_during_run"] if c["maybe_run_code"]]
    rows = [
        ("machine", host),
        ("utilisateur", UNKNOWN if imported else run.path.owner()),
        ("lancement", run.launch.isoformat(sep=" ", timespec="seconds")),
        ("commit au lancement", describe(context["commit_at_launch"])),
        ("commits ≤ 90 min après", ", ".join(f"`{c['hash']}`" for c in candidates) or "aucun"),
        ("diff au lancement", UNKNOWN if imported else launch_diff(run, context, folder)),
        ("données brutes", f"`outputs/{run.name}` sur {host}"),
    ]
    lines = [f"# {run.name}", "", "| | |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += ["", "## Faits établis", "", ""]
    (folder / "run.md").write_text("\n".join(lines))
    return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--host")
    args = parser.parse_args()
    run = load_run(args.run)
    host = args.host or socket.gethostname()
    folder = knowledge_dir(run, host) or create(run, host, imported=args.host is not None)
    print(folder.relative_to(REPO))


if __name__ == "__main__":
    main()
