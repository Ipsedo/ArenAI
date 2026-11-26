import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import arenai_run as ar  # noqa: E402

NAN = float("nan")
LAUNCH = datetime(2026, 11, 20, 10, 0)


def make_run(rows, config=None, columns=None, number=7, launch=LAUNCH, last_write=None):
    return ar.Run(
        path=Path(f"train_{number}_ppo"), number=number, config=config or {},
        columns=columns or (list(rows[0]) if rows else []), rows=rows, launch=launch,
        last_write=last_write or launch + timedelta(hours=2),
    )


def is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def commit_line(name, minutes_after_launch):
    date = LAUNCH + timedelta(minutes=minutes_after_launch)
    return f"{name}|{date:%Y-%m-%d %H:%M:%S} +0000|subject {name}"


def fake_git(at_launch="", during=""):
    def git(*args):
        if args[0] == "log" and "-1" in args:
            return at_launch
        if args[0] == "log":
            return during
        return ""
    return git


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "OUTPUTS", tmp_path)
    monkeypatch.setattr(ar, "git", fake_git())
    return tmp_path


# loading

def write_run(folder, csv_text, config=None):
    folder.mkdir()
    (folder / "metrics.csv").write_text(csv_text)
    if config is not None:
        (folder / "config.json").write_text(json.dumps(config))
    return folder


CSV = "hit;D;index;\n0.5;250;0;\n0.1;250;30;\nbad;250;60;\n0.3;300\n0.4;300;90;\n"


def test_load_run_ignores_index_zero_and_malformed_rows(outputs):
    run = ar.load_run(write_run(outputs / "train_12_ppg", CSV))
    assert run.columns == ["hit", "D", "index"]
    assert [r["index"] for r in run.rows] == [30.0, 60.0, 90.0]
    assert is_nan(run.rows[1]["hit"])
    assert run.col("hit") == [0.1, 0.4]
    assert run.col("absent") == []
    assert (run.number, run.name, run.config) == (12, "train_12_ppg", {})


def test_load_run_reads_config_and_dates_launch_from_config_mtime(outputs):
    folder = write_run(outputs / "train_12_ppg", CSV, {"agent": {"epochs": 2}})
    os.utime(folder / "config.json", (LAUNCH.timestamp(), LAUNCH.timestamp()))
    last = LAUNCH + timedelta(hours=3)
    os.utime(folder / "metrics.csv", (last.timestamp(), last.timestamp()))
    run = ar.load_run(folder)
    assert run.config == {"agent": {"epochs": 2}}
    assert (run.launch, run.last_write) == (LAUNCH, last)


def test_resolve_run_by_number(outputs):
    folder = write_run(outputs / "train_12_ppg", CSV)
    write_run(outputs / "train_120_ppg", CSV)
    assert ar.resolve_run("12") == folder.resolve()
    assert ar.resolve_run("train_12") == folder.resolve()
    with pytest.raises(SystemExit):
        ar.resolve_run("13")


def test_run_number():
    assert ar.run_number(Path("outputs/train_504_ppo_liquid")) == 504
    assert ar.run_number(Path("outputs/comparisons")) == -1
    assert ar.run_number(Path("outputs/train_x_ppo")) == -1


def test_previous_run_is_the_highest_lower_number_with_config_and_csv(outputs):
    write_run(outputs / "train_5_ppo", CSV, {})
    write_run(outputs / "train_6_ppo", CSV)
    write_run(outputs / "train_9_ppo", CSV, {})
    assert ar.previous_run(make_run([], number=7)) == outputs / "train_5_ppo"
    assert ar.previous_run(make_run([], number=5)) is None


# config

def test_flatten_and_config_value():
    run = make_run([], {"agent": {"lr": 1e-4, "net": {"hidden": 64}},
                        "environment": {"lr": 9, "nb_tanks": 16}, "train": {"save_every": 1000}})
    assert ar.flatten(run.config) == {
        "agent.lr": 1e-4, "agent.net.hidden": 64, "environment.lr": 9,
        "environment.nb_tanks": 16, "train.save_every": 1000}
    assert ar.config_value(run, "lr") == 1e-4
    assert ar.config_value(run, "nb_tanks") == 16
    assert ar.config_value(run, "save_every") == 1000
    assert ar.config_value(run, "hidden") is None


def test_config_diff_lists_changed_added_removed_keys_except_output_folder():
    a = make_run([], {"agent": {"lr": 1e-4, "epochs": 2}, "train": {"output_folder": "a"}})
    b = make_run([], {"agent": {"lr": 3e-4, "epochs": 2, "kl": 0.02},
                      "train": {"output_folder": "b"}})
    assert ar.config_diff(a, b) == {"agent.lr": (1e-4, 3e-4), "agent.kl": (None, 0.02)}
    assert ar.config_diff(b, a) == {"agent.lr": (3e-4, 1e-4), "agent.kl": (0.02, None)}
    assert ar.config_diff(a, a) == {}


# context

def test_throughput_in_steps_and_tank_steps_per_hour(outputs):
    run = make_run([{"index": 3000.0}, {"index": 6000.0}], {"environment": {"nb_tanks": 16}})
    ctx = ar.build_context(run)
    assert ctx["hours"] == 2.0
    assert ctx["steps_per_hour"] == 3000
    assert ctx["tank_steps_per_hour"] == 48000
    assert ctx["running"] is False
    assert ctx["commit_at_launch"] is None and ctx["effective_commit"] is None
    assert ctx["commits_during_run"] == [] and ctx["next_run_launch"] is None


def test_running_when_csv_written_in_the_last_15_minutes(outputs):
    now = datetime.now()
    recent = make_run([{"index": 30.0}], launch=now - timedelta(hours=1),
                      last_write=now - timedelta(minutes=14))
    stale = make_run([{"index": 30.0}], launch=now - timedelta(hours=1),
                     last_write=now - timedelta(minutes=16))
    assert ar.build_context(recent)["running"] is True
    assert ar.build_context(stale)["running"] is False


def test_effective_commit_is_the_last_commit_within_90_minutes_of_launch(outputs, monkeypatch):
    during = "\n".join(commit_line(n, m) for n, m in (("b", 30), ("c", 90), ("d", 91)))
    monkeypatch.setattr(ar, "git", fake_git(commit_line("a", -60), during))
    ctx = ar.build_context(make_run([{"index": 30.0}]))
    assert ctx["commit_at_launch"]["hash"] == "a"
    assert [c["hash"] for c in ctx["commits_during_run"]] == ["b", "c", "d"]
    assert [c["minutes_after_launch"] for c in ctx["commits_during_run"]] == [30, 90, 91]
    assert [c["maybe_run_code"] for c in ctx["commits_during_run"]] == [True, True, False]
    assert ctx["effective_commit"]["hash"] == "c"


def test_effective_commit_is_the_launch_commit_without_candidate(outputs, monkeypatch):
    monkeypatch.setattr(ar, "git", fake_git(commit_line("a", -60), commit_line("d", 120)))
    ctx = ar.build_context(make_run([{"index": 30.0}]))
    assert ctx["effective_commit"]["hash"] == "a"


def test_commits_after_the_next_run_launch_are_not_candidates(outputs, monkeypatch):
    nxt = LAUNCH + timedelta(minutes=45)
    folder = write_run(outputs / "train_8_ppo", CSV, {})
    os.utime(folder / "config.json", (nxt.timestamp(), nxt.timestamp()))
    during = "\n".join(commit_line(n, m) for n, m in (("b", 30), ("c", 60)))
    monkeypatch.setattr(ar, "git", fake_git(commit_line("a", -60), during))
    ctx = ar.build_context(make_run([{"index": 30.0}], number=7))
    assert ctx["next_run_launch"] == nxt.isoformat(timespec="minutes")
    assert [c["maybe_run_code"] for c in ctx["commits_during_run"]] == [True, False]
    assert ctx["effective_commit"]["hash"] == "b"


# code diff

DIFF = """diff --git a/src/a.cpp b/src/a.cpp
index 1..2 100644
--- a/src/a.cpp
+++ b/src/a.cpp
@@ -1,2 +1,4 @@
-int x = 1;
-    // old comment
+int x = 2;
+int y = 3;
+  // new comment
+
diff --git a/src/b.h b/src/b.h
--- a/src/b.h
+++ b/src/b.h
@@ -1 +1 @@
-// before
+// after
"""


def test_substantive_diff_counts_lines_that_are_neither_comment_nor_blank(monkeypatch):
    monkeypatch.setattr(ar, "git", lambda *args: DIFF)
    diff = ar.substantive_diff("a", "b")
    assert diff == {"stat": "src/a.cpp  +2 -1", "comment_only_files": 1}


def test_substantive_diff_without_change(monkeypatch):
    monkeypatch.setattr(ar, "git", lambda *args: "")
    assert ar.substantive_diff("a", "b") == {"stat": "", "comment_only_files": 0}


def test_code_diff_spans_the_effective_commits(monkeypatch):
    calls = []

    def git(*args):
        calls.append(args)
        return "c1 date subject" if args[0] == "log" else DIFF

    monkeypatch.setattr(ar, "git", git)
    a, b = make_run([]), make_run([])
    a.context = {"effective_commit": {"hash": "aaa"}}
    b.context = {"effective_commit": {"hash": "bbb"}}
    diff = ar.code_diff(a, b)
    assert (diff["from"], diff["to"], diff["log"]) == ("aaa", "bbb", ["c1 date subject"])
    assert diff["stat"] == "src/a.cpp  +2 -1"
    assert calls[0][-1] == "aaa..bbb"
    assert calls[1][:5] == ("diff", "-U0", "--no-color", "aaa", "bbb")


def test_code_diff_is_empty_without_effective_commit():
    a, b = make_run([]), make_run([])
    a.context = {"effective_commit": None}
    b.context = {"effective_commit": {"hash": "bbb"}}
    assert ar.code_diff(a, b) == {}


# CSV statistics

def test_mean_ignores_nan_and_is_nan_when_empty():
    assert ar.mean([1.0, NAN, 2.0]) == 1.5
    assert is_nan(ar.mean([]))
    assert is_nan(ar.mean([NAN]))


D_ROWS = [
    {"index": 30.0, "D": 250.0, "hit": 0.1},
    {"index": 60.0, "D": 250.0, "hit": 0.3},
    {"index": 90.0, "D": 300.0, "hit": 0.2},
    {"index": 120.0, "D": 250.0, "hit": NAN},
    {"index": 150.0, "D": NAN, "hit": 0.9},
    {"index": 180.0, "D": 350.0, "hit": 0.4},
]


def test_d_milestones_are_first_appearances():
    assert ar.d_milestones(make_run(D_ROWS)) == [(250.0, 30.0), (300.0, 90.0), (350.0, 180.0)]


def test_d_milestones_without_d_column():
    assert ar.d_milestones(make_run([{"index": 30.0, "hit": 0.1}])) == []


def test_per_d_level_means_over_every_row_of_the_level():
    levels = ar.per_d_level(make_run(D_ROWS), ["hit", "absent"])
    assert list(levels) == [250.0, 300.0, 350.0]
    assert levels[250.0]["n"] == 3
    assert (levels[250.0]["from"], levels[250.0]["to"]) == (30.0, 120.0)
    assert levels[250.0]["hit"] == pytest.approx(0.2)
    assert levels[300.0]["hit"] == 0.2 and levels[350.0]["hit"] == 0.4
    assert is_nan(levels[250.0]["absent"])


def test_windows_are_ten_index_slices_open_left_closed_right():
    rows = [
        {"index": 50.0, "D": 250.0, "hit": 0.1},
        {"index": 100.0, "D": 300.0, "hit": 0.3},
        {"index": 101.0, "D": 300.0, "hit": 0.5},
        {"index": 1000.0, "D": 400.0, "hit": 0.7},
    ]
    w = ar.windows(make_run(rows), ["hit"])
    assert [(x["from"], x["to"], x["n"]) for x in w] == [(0, 100, 2), (100, 200, 1), (900, 1000, 1)]
    assert [x["D"] for x in w] == [300.0, 300.0, 400.0]
    assert [x["hit"] for x in w] == [pytest.approx(0.2), 0.5, 0.7]


def test_windows_of_an_empty_run():
    assert ar.windows(make_run([]), ["hit"]) == []


def test_at_index_averages_the_rows_in_the_half_open_window():
    rows = [
        {"index": 85.0, "D": 250.0, "hit": 1.0},
        {"index": 86.0, "D": 250.0, "hit": 0.1},
        {"index": 100.0, "D": 300.0, "hit": NAN},
        {"index": 115.0, "D": 350.0, "hit": 0.3},
        {"index": 116.0, "D": 400.0, "hit": 1.0},
    ]
    stats = ar.at_index(make_run(rows), ["hit"], 100, 15)
    assert stats["n"] == 3
    assert stats["D"] == 350.0
    assert stats["hit"] == pytest.approx(0.2)
    assert ar.at_index(make_run(rows), ["hit"], 500, 15) is None


def test_extremes_give_value_index_and_d():
    e = ar.extremes(make_run(D_ROWS), ["hit", "absent"])
    assert list(e) == ["hit"]
    assert e["hit"]["max"][:2] == (0.9, 150.0) and is_nan(e["hit"]["max"][2])
    assert e["hit"]["min"] == (0.1, 30.0, 250.0)
    assert e["hit"]["last"] == (0.4, 180.0, 350.0)


def test_alpha_columns():
    run = make_run([], columns=["hit", "alpha_fire", "Hc", "alpha_zoom"])
    assert ar.alpha_columns(run) == ["alpha_fire", "alpha_zoom"]


# signals

CONFIG = {"agent": {"target_kl": 0.02, "initial_alpha": 0.001},
          "environment": {"final_spawn_width": 1000}}


def healthy_rows():
    return [
        {"index": 100.0 * (i + 1), "ev": 0.5, "skip": 0.0, "kl": 0.01, "Hc": 0.5,
         "alpha_fire": 0.001, "fire": 0.5, "hit": 0.1, "kill": 1.0,
         "D": 250.0 if i < 8 else 300.0}
        for i in range(10)
    ]


def signals_with(**tail_overrides):
    rows = healthy_rows()
    for r in rows[8:]:
        r.update(tail_overrides)
    return ar.signals(make_run(rows, CONFIG))


def test_healthy_run_raises_no_signal():
    assert ar.signals(make_run(healthy_rows(), CONFIG)) == []


def test_fewer_than_five_rows_is_not_interpreted():
    out = ar.signals(make_run(healthy_rows()[:4], CONFIG))
    assert len(out) == 1 and "moins de 5 lignes" in out[0]


def test_signal_weak_critic_when_tail_ev_under_0_2():
    out = signals_with(ev=0.19)
    assert len(out) == 1 and "critic faible" in out[0]
    assert signals_with(ev=0.21) == []


def test_signal_negative_ev_on_more_than_20_percent_of_rows():
    rows = healthy_rows()
    for r in rows[:3]:
        r["ev"] = -0.1
    out = ar.signals(make_run(rows, CONFIG))
    assert len(out) == 1 and "ev < 0 sur 30%" in out[0]
    rows[2]["ev"] = 0.5
    assert ar.signals(make_run(rows, CONFIG)) == []


def test_signal_kl_regime_when_tail_skip_above_0_1():
    out = signals_with(skip=0.11)
    assert len(out) == 1 and "régime KL" in out[0]
    assert signals_with(skip=0.09) == []


def test_signal_tail_kl_above_target():
    out = signals_with(kl=0.03)
    assert len(out) == 1 and "target_kl" in out[0]


def test_signal_sigma_collapse_when_tail_hc_negative():
    out = signals_with(Hc=-0.1)
    assert len(out) == 1 and out[0].startswith("Hc")


def test_signal_alpha_off_its_floor_above_1_5_times_initial():
    rows = healthy_rows()
    rows[4]["alpha_fire"] = 0.0016
    out = ar.signals(make_run(rows, CONFIG))
    assert len(out) == 1 and out[0].startswith("alpha_fire a quitté le plancher")
    rows[4]["alpha_fire"] = 0.0014
    assert ar.signals(make_run(rows, CONFIG)) == []


def test_signal_trigger_off_when_fire_ends_under_0_05_after_a_peak():
    out = signals_with(fire=0.04)
    assert len(out) == 1 and "gâchette éteinte" in out[0]
    assert signals_with(fire=0.06) == []


def test_signal_hit_and_kill_under_70_percent_of_their_peak():
    out = signals_with(hit=0.069)
    assert len(out) == 1 and out[0].startswith("hit final") and "70 % du pic" in out[0]
    out = signals_with(kill=0.69)
    assert len(out) == 1 and out[0].startswith("kill final")
    assert signals_with(hit=0.071, kill=0.71) == []


def test_signal_curriculum_stuck_below_final_width_for_over_30_percent_of_the_run():
    rows = healthy_rows()
    for r in rows:
        r["D"] = 250.0
    out = ar.signals(make_run(rows, CONFIG))
    assert len(out) == 1 and out[0].startswith("D bloqué à 250")
    for r in rows:
        r["D"] = 1000.0
    assert ar.signals(make_run(rows, CONFIG)) == []


# summary

def test_summary_gathers_every_table(outputs):
    run = make_run(D_ROWS, {"agent": {"lr": 1e-4}})
    s = ar.summary(run)
    assert (s["run"], s["nb_rows"], s["last_index"]) == ("train_7_ppo", 6, 180.0)
    assert s["d_milestones"] == ar.d_milestones(run)
    assert s["per_d_level"][250.0]["hit"] == pytest.approx(0.2)
    assert s["extremes"]["hit"]["last"] == (0.4, 180.0, 350.0)
    assert s["windows"][-1]["to"] == 180.0
    assert s["signals"] == ar.signals(run)
    assert s["context"]["steps_per_hour"] == 90


# formatting

def test_fmt():
    assert ar.fmt(NAN).strip() == "-"
    assert ar.fmt(None).strip() == "-"
    assert ar.fmt(0.0).strip() == "0"
    assert ar.fmt(250.0).strip() == "250"
    assert ar.fmt(1234.4).strip() == "1234"
    assert ar.fmt(12.34).strip() == "12.3"
    assert ar.fmt(0.1234).strip() == "0.123"
    assert ar.fmt(0.001234).strip() == "1.2e-03"
    assert ar.fmt("save_3") == "  save_3"
    assert ar.fmt_value(0.00012345) == "0.0001234"
    assert ar.fmt_value("ppo") == "ppo"


def test_table_aligns_columns():
    text = ar.table(["D", "hit"], [[250.0, 0.1], [1000.0, NAN]], width=6)
    assert text.splitlines() == ["D       hit", "250   0.100", "1000      -"]


def test_knowledge_dir_matches_host_number_and_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "KNOWLEDGE_RUNS", tmp_path)
    mine = tmp_path / "2026-11-23-22-49-10_here_train_7_ppo_liquid"
    for folder in (mine, tmp_path / "2026-11-20-10-00-00_there_train_7_ppo_liquid",
                   tmp_path / "2026-11-21-10-00-00_here_train_17_ppo_liquid"):
        folder.mkdir()
    run = ar.Run(path=Path("outputs/train_7_ppo_liquid"), number=7, config={}, columns=[],
                 rows=[], launch=LAUNCH, last_write=LAUNCH)
    assert ar.knowledge_dir(run, "here") == mine
    assert ar.knowledge_dir(run, "elsewhere") is None
