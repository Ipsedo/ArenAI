import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import probe_timeline as pt  # noqa: E402

import arenai_run as ar  # noqa: E402

CSV = (
    "hit;D;alpha_fire;alpha_zoom;index\n"
    "0.9;250;0.001;0.001;1000\n"
    "0.2;300;0.002;0.004;2000\n"
    "0.4;350;0.002;0.006;2400\n"
    "0.9;400;0.009;0.001;3000\n"
)


def metrics(ratio):
    head = {"p10": 0.1, "p50": 0.2, "p90": 0.3, "at_ceiling": 0.0}
    mu = {"mean": 0.0, "abs_p50": 0.5, "saturated": 0.0}
    return {
        "trigger": {"ratio": ratio, "p_fire_aligned_close": 0.5, "p_fire_off_axis": 0.25,
                    "p_fire_armed": 0.3, "n_aligned_close": 400, "n_off_axis": 900},
        "shots": {"per_tank_second": 0.5, "landed": 120.0, "hit_per_shell": 0.25},
        "reserve": {"first_empty_s_p50": 6.5, "emptied": 0.75, "disarmed_fraction": 0.125},
        "turret": {"below_5deg": 0.15},
        "heads": {"aim_dims": ["canon_x"], "all_dims": ["canon_x"], "sigma_canon_x": head,
                  "mu_canon_x": mu, "p_zoom": 0.5},
    }


def write_probe(folder, save, side, ratio):
    meta = {"save": f"save_{save}", "save_index": save, "index": save * 1000, "side": side,
            "deterministic": False, "episodes": 4}
    (folder / "probes" / f"save_{save}_side{side}.json").write_text(
        json.dumps({"meta": meta, "metrics": metrics(ratio)}))


@pytest.fixture
def report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ar, "OUTPUTS", tmp_path)
    monkeypatch.setattr(ar, "git", lambda *args: "")
    folder = tmp_path / "train_7_ppo"
    for name in ("probes", "save_0", "save_1", "save_2", "save_3"):
        (folder / name).mkdir(parents=True)
    (folder / "metrics.csv").write_text(CSV)
    (folder / "config.json").write_text(json.dumps({"train": {"save_every": 1000}}))
    write_probe(folder, 3, 550, 6.0)
    write_probe(folder, 2, 550, 4.0)
    write_probe(folder, 2, 1000, 2.0)
    (folder / "probes" / "other.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["probe_timeline.py", str(folder), "--half-width", "500"])
    pt.main()
    return capsys.readouterr().out


def section(text, title):
    return [line.split() for line in text.split(title)[1].split("\n## ")[0].splitlines() if line]


def test_one_table_per_side_ordered_by_save(report):
    rows = section(report, "## Côté 550, 4 épisodes")
    assert rows[1][:6] == ["save_2", "2k", "4.000", "0.500", "0.250", "0.300"]
    assert rows[2][:3] == ["save_3", "3k", "6.000"]
    assert section(report, "## Côté 1000, 4 épisodes")[1][:3] == ["save_2", "2k", "2.000"]


def test_headline_columns_follow_probe_metrics(report):
    header = section(report, "## Côté 550, 4 épisodes")[0]
    assert header[:5] == ["save", "index", "ratio", "p_al", "p_off"]
    assert header[-1] == "p_zoom"


def test_csv_around_a_save_reads_d_at_the_save_and_means_in_the_window(report):
    rows = section(report, "## CSV autour de chaque save (±0k)")
    assert rows[0] == ["save", "index", "D", "hit", "α_max", "α_argmax"]
    assert rows[1] == ["save_2", "2k", "300", "0.300", "5.0e-03", "zoom"]
    assert rows[2] == ["save_3", "3k", "400", "0.900", "9.0e-03", "fire"]


def test_unprobed_saves_are_listed(report):
    assert "saves non sondées : [1]" in report
