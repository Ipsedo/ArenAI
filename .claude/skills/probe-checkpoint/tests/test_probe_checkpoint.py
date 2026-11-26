import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import arenai_run as ar  # noqa: E402
import probe_checkpoint as pc  # noqa: E402
from test_probe_metrics import DECISION, write_csv  # noqa: E402

CSV = "hit;D;index\n0.1;250;1000\n0.2;300;2000\n0.3;350;3000\n"


def fake_git(*args):
    if args[0] == "log" and "-1" in args:
        return "abc1234|2026-11-01 10:00:00 +0000|subject"
    return ""


@pytest.fixture
def run_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "OUTPUTS", tmp_path)
    monkeypatch.setattr(ar, "git", fake_git)
    monkeypatch.setattr(pc, "git", fake_git)
    folder = tmp_path / "train_7_ppo"
    (folder / "save_2").mkdir(parents=True)
    (folder / "save_2" / "actor.pt").touch()
    (folder / "probes").mkdir()
    (folder / "metrics.csv").write_text(CSV)
    (tmp_path / "arenai_agent_probe").touch()
    rows = [{"step": s, "shells": 0 if s == 3 else 0.5} for s in range(12)]
    write_csv(folder / "probes" / "save_2_side550.csv", DECISION, rows)
    return folder


def analyze_only(folder, monkeypatch, config):
    (folder / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", [
        "probe_checkpoint.py", str(folder / "save_2"), "--analyze-only", "--sides", "550",
        "--binary", str(folder.parent / "arenai_agent_probe")])
    pc.main()
    return json.loads((folder / "probes" / "save_2_side550.json").read_text())


def test_step_seconds_is_action_repeat_times_wanted_frequency(run_folder, monkeypatch):
    config = {"agent": {"action_repeat": 4}, "environment": {"wanted_frequency": 0.05},
              "train": {"save_every": 1000}}
    result = analyze_only(run_folder, monkeypatch, config)
    assert result["metrics"]["step_seconds"] == pytest.approx(0.2)
    assert result["metrics"]["reserve"]["first_empty_s_p50"] == pytest.approx(0.6)


def test_step_seconds_defaults_to_a_tenth_of_a_second(run_folder, monkeypatch):
    result = analyze_only(run_folder, monkeypatch, {"train": {"save_every": 1000}})
    assert result["metrics"]["step_seconds"] == pytest.approx(0.1)
    assert result["metrics"]["reserve"]["first_empty_s_p50"] == pytest.approx(0.3)


def test_meta_places_the_save_at_k_times_save_every(run_folder, monkeypatch):
    meta = analyze_only(run_folder, monkeypatch, {"train": {"save_every": 1000}})["meta"]
    assert (meta["save"], meta["save_index"], meta["index"]) == ("save_2", 2, 2000)
    assert meta["D_at_save"] == 300.0
    assert (meta["side"], meta["episodes"], meta["deterministic"]) == (550.0, 4, False)
    assert meta["inference_diff"] == ""


def test_resolve_save(run_folder):
    run, save = pc.resolve_save(str(run_folder / "save_2"), None)
    assert (run.name, save) == ("train_7_ppo", "save_2")
    assert pc.resolve_save(str(run_folder), 2)[1] == "save_2"
    assert pc.resolve_save(str(run_folder), "save_2")[1] == "save_2"
    with pytest.raises(SystemExit):
        pc.resolve_save(str(run_folder), None)


def test_repr_signature_keeps_the_head_layers(tmp_path):
    (tmp_path / "actor_repr.txt").write_text(
        "Actor(\n"
        "  (cnn): Sequential(\n"
        "    0: Conv2d(3, 16, kernel_size=3)\n"
        "    1: GroupNorm(4, 16)\n"
        "    2: Linear(in=64, out=32)\n"
        "  )\n"
        "  mu: Linear(in=32, out=4)\n"
        "  sigma: SoftClamp(min=0.01, max=0.7)\n"
        ")\n")
    assert pc.repr_signature(tmp_path) == ["2: Linear", "mu: Linear", "sigma: SoftClamp"]
    assert pc.repr_signature(tmp_path / "absent") == []
