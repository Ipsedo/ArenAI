import csv
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import match_metrics as mm  # noqa: E402

TANK = {
    "match": "0", "second": "1", "tank": "0", "alive": "1", "out_of_shells": "0", "fired": "0",
    "hits": "0", "kills": "0", "canon_elevation": "0", "best_angle": "180",
}
PLAYER = {"match": "0", "second": "1", "alive": "1", "impacts": "0"}
INFO = {"match": "0", "seconds": "300", "end": "timeout", "enemies_armed": "0"}

FOUND = "found: armed within <5° of a target"
FIRST = "found: 1st target <5° (s, median)"
DRY_TIME = "dangerous: time alive dry"
ARMED_END = "dangerous: armed at end / tanks"


def tank(**kw):
    return {**TANK, **{k: str(v) for k, v in kw.items()}}


def player(**kw):
    return {**PLAYER, **{k: str(v) for k, v in kw.items()}}


def info(**kw):
    return {**INFO, **{k: str(v) for k, v in kw.items()}}


def dry_at(t):
    return f"dangerous: alive dry at {t} s"


def is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


# match header

def test_duration_and_all_dry_end():
    m = mm.one_match([tank()], [player()], info(seconds=42.5, end="all_dry"))
    assert m["duration_s"] == 42.5
    assert m["end_dry"] == 1.0
    assert mm.one_match([tank()], [player()], info(end="timeout"))["end_dry"] == 0.0


# 1. finds the enemies

def test_found_is_share_of_armed_living_rows_under_5_degrees():
    m = mm.one_match([
        tank(best_angle=4.99),
        tank(best_angle=5.0),
        tank(best_angle=0, out_of_shells=1),
        tank(best_angle=0, alive=0),
    ], [player()], info())
    assert m[FOUND] == pytest.approx(1 / 2)


def test_found_is_nan_without_armed_living_row():
    m = mm.one_match([tank(out_of_shells=1, best_angle=0), tank(alive=0, best_angle=0)],
                     [player()], info())
    assert is_nan(m[FOUND])


def test_first_target_is_median_over_tanks_of_first_aligned_second_after_1s():
    m = mm.one_match([
        tank(tank=0, second=0.5, best_angle=1),
        tank(tank=0, second=1.0, best_angle=1),
        tank(tank=0, second=2.0, best_angle=3),
        tank(tank=0, second=2.5, best_angle=3),
        tank(tank=1, second=3.0, best_angle=0, alive=0),
        tank(tank=1, second=6.0, best_angle=4, out_of_shells=1),
        tank(tank=2, second=4.0, best_angle=5.0),
    ], [player()], info())
    assert m[FIRST] == pytest.approx(4.0)


def test_first_target_is_nan_when_no_tank_aligns():
    m = mm.one_match([tank(second=2.0, best_angle=30)], [player()], info())
    assert is_nan(m[FIRST])


# 2. reste dangereux

def test_dry_at_time_is_share_of_living_tanks_with_empty_reserve():
    m = mm.one_match([
        tank(second=1, tank=0),
        tank(second=10, tank=0, out_of_shells=1),
        tank(second=10, tank=1),
        tank(second=10, tank=2, alive=0, out_of_shells=1),
        tank(second=29.5, tank=0),
        tank(second=30.4, tank=1, out_of_shells=1),
    ], [player()], info())
    assert m[dry_at(10)] == pytest.approx(1 / 2)
    assert m[dry_at(30)] == 1.0
    assert is_nan(m[dry_at(60)])
    assert is_nan(m[dry_at(120)])


def test_dry_at_time_is_nan_when_every_tank_is_dead():
    m = mm.one_match([tank(second=1), tank(second=10, alive=0, out_of_shells=1)],
                     [player()], info())
    assert is_nan(m[dry_at(10)])


def test_living_time_dry_is_share_of_living_rows_out_of_shells():
    m = mm.one_match([
        tank(second=1), tank(second=2), tank(second=3, out_of_shells=1),
        tank(second=4, out_of_shells=1, alive=0),
    ], [player()], info())
    assert m[DRY_TIME] == pytest.approx(1 / 3)


def test_living_time_dry_is_zero_for_a_tank_that_never_empties():
    m = mm.one_match([tank(second=1), tank(second=2)], [player()], info())
    assert m[DRY_TIME] == 0.0


def test_living_time_dry_is_nan_without_living_row():
    m = mm.one_match([tank(alive=0)], [player()], info())
    assert is_nan(m[DRY_TIME])


def test_armed_at_end_is_per_distinct_tank():
    rows = [tank(tank=i, second=s) for i in range(4) for s in (1, 2)]
    m = mm.one_match(rows, [player()], info(enemies_armed=3))
    assert m[ARMED_END] == pytest.approx(3 / 4)


# 3. hits the player

def test_player_impacts_first_impact_and_death():
    m = mm.one_match([tank()], [
        player(second=1), player(second=2, impacts=2), player(second=3),
        player(second=4, impacts=1, alive=0), player(second=5, alive=0),
    ], info())
    assert m["player: impacts received"] == 3
    assert m["player: 1st impact (s)"] == 2.0
    assert m["player: killed"] == 1.0
    assert m["player: died at (s)"] == 4.0


def test_player_untouched():
    m = mm.one_match([tank()], [player(second=1), player(second=2)], info())
    assert m["player: impacts received"] == 0
    assert is_nan(m["player: 1st impact (s)"])
    assert m["player: killed"] == 0.0
    assert is_nan(m["player: died at (s)"])


# 4. hits and kills the others

def test_combat_counts_are_summed_over_rows_and_divided_by_tanks():
    m = mm.one_match([
        tank(tank=0, second=1, fired=2, hits=1, kills=1),
        tank(tank=0, second=2, fired=1),
        tank(tank=1, second=1, fired=1),
        tank(tank=1, second=2, alive=0),
        tank(tank=2, second=1),
    ], [player()], info())
    assert m["combat: touches / tank"] == pytest.approx(1 / 3)
    assert m["combat: kills / tank"] == pytest.approx(1 / 3)
    assert m["combat: shells fired / tank"] == pytest.approx(4 / 3)
    assert m["combat: hit / shell"] == pytest.approx(1 / 4)
    assert m["combat: tanks without a hit"] == pytest.approx(2 / 3)


def test_hit_per_shell_is_nan_without_shot():
    m = mm.one_match([tank()], [player()], info())
    assert is_nan(m["combat: hit / shell"])
    assert m["combat: tanks without a hit"] == 1.0


# 5. tenue visible

def test_canon_elevation_armed_and_dry():
    m = mm.one_match([
        tank(canon_elevation=-10), tank(canon_elevation=-5), tank(canon_elevation=2),
        tank(canon_elevation=-20, out_of_shells=1), tank(canon_elevation=-30, out_of_shells=1),
        tank(canon_elevation=-80, alive=0), tank(canon_elevation=-80, alive=0, out_of_shells=1),
    ], [player()], info())
    assert m["hold: gun elevation armed (°, median)"] == -5.0
    assert m["hold: gun elevation dry (°, median)"] == -25.0
    assert m["hold: armed gun < -5°"] == pytest.approx(1 / 3)


def test_canon_elevation_is_nan_for_an_empty_population():
    armed_only = mm.one_match([tank(canon_elevation=1)], [player()], info())
    assert is_nan(armed_only["hold: gun elevation dry (°, median)"])
    dry_only = mm.one_match([tank(canon_elevation=1, out_of_shells=1)], [player()], info())
    assert is_nan(dry_only["hold: gun elevation armed (°, median)"])
    assert is_nan(dry_only["hold: armed gun < -5°"])


# files and summary

def write_checkpoint(prefix, best_angles_by_match):
    tanks, players, matches = [], [], []
    for match, angles in best_angles_by_match.items():
        tanks += [tank(match=match, second=s + 1, best_angle=a) for s, a in enumerate(angles)]
        players.append(player(match=match))
        matches.append(info(match=match, seconds=100 * (match + 1)))
    write_csv(f"{prefix}_tanks.csv", tanks)
    write_csv(f"{prefix}_player.csv", players)
    write_csv(f"{prefix}_matches.csv", matches)


def test_checkpoint_groups_rows_by_match(tmp_path):
    prefix = str(tmp_path / "save_3_side500")
    write_checkpoint(prefix, {0: [1, 1, 30, 30], 1: [30, 30, 30, 30]})
    first, second = mm.checkpoint(prefix)
    assert (first["duration_s"], second["duration_s"]) == (100.0, 200.0)
    assert first[FOUND] == pytest.approx(1 / 2)
    assert second[FOUND] == 0.0
    assert first[FIRST] == 2.0
    assert is_nan(second[FIRST])


def test_main_reports_mean_sample_sd_and_count_over_matches(tmp_path, monkeypatch, capsys):
    new, ref = str(tmp_path / "new"), str(tmp_path / "ref")
    write_checkpoint(new, {0: [1, 1, 30, 30], 1: [30, 30, 30, 30], 2: [1, 1, 1, 30]})
    write_checkpoint(ref, {0: [30, 1, 1, 1]})
    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["match_metrics.py", f"new={new}", f"ref={ref}",
                                      "--json", str(out)])
    mm.main()
    summary = json.loads(out.read_text())
    assert list(summary) == ["new", "ref"]
    found = summary["new"][FOUND]
    assert found["n"] == 3
    assert found["mean"] == pytest.approx((0.5 + 0.0 + 0.75) / 3)
    assert found["sd"] == pytest.approx(math.sqrt((0.0069444444 + 0.1736111111 + 0.1111111111) / 2))
    first = summary["new"][FIRST]
    assert (first["n"], first["mean"], first["sd"]) == (2, 2.0, 0.0)
    assert summary["ref"][FOUND] == {"mean": 0.75, "sd": 0.0, "n": 1}
    assert "matchs" in capsys.readouterr().out
