import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compare_runs as cr  # noqa: E402

import arenai_run as ar  # noqa: E402

REFERENCE = [(90, 0.1, 250), (100, 0.3, 250), (115, 0.2, 250), (116, 0.9, 250), (200, 0.5, 300)]
CANDIDATE = [(100, 0.4, 250), (200, 0.6, 300), (300, 0.7, 300)]


def write_run(folder, rows):
    folder.mkdir()
    lines = ["hit;D;index"] + [f"{hit};{d};{index * 1000}" for index, hit, d in rows]
    (folder / "metrics.csv").write_text("\n".join(lines) + "\n")
    return str(folder)


def table(text, section, column):
    body = text.split(section)[1].split("\n## ")[0]
    return [line.split() for line in body.split(f"\n{column}\n")[1].split("\n\n")[0].splitlines()]


@pytest.fixture
def report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ar, "OUTPUTS", tmp_path)
    monkeypatch.setattr(ar, "git", lambda *args: "")
    runs = [write_run(tmp_path / "train_7_ppo", REFERENCE),
            write_run(tmp_path / "train_8_ppo", CANDIDATE)]
    monkeypatch.setattr(sys, "argv", ["compare_runs.py"] + runs)
    cr.main()
    return capsys.readouterr().out


def test_d_milestones_side_by_side(report):
    body = report.split("## D milestones")[1].split("\n## ")[0].splitlines()[1:]
    assert [line.split() for line in body if line] == [
        ["D", "7", "8"], ["250", "90k", "100k"], ["300", "200k", "200k"]]


def test_equal_d_means_cover_every_row_of_the_level(report):
    assert table(report, "## At equal D", "hit") == [
        ["D", "7", "8"], ["250", "0.375", "0.400"], ["300", "0.500", "0.650"]]
    assert table(report, "## At equal D", "n") == [["D", "7", "8"], ["250", "4", "1"], ["300", "1", "2"]]


def test_equal_index_points_stop_at_the_least_advanced_run(report):
    assert table(report, "## At equal index", "hit") == [
        ["index", "7", "8"], ["100k", "0.200", "0.400"], ["200k", "0.500", "0.600"]]
    assert table(report, "## At equal index", "D") == [
        ["index", "7", "8"], ["100k", "250", "250"], ["200k", "300", "300"]]


def test_peaks_give_value_index_and_d(report):
    assert "hit max | 7: 0.900 @ 116k D250 | 8: 0.700 @ 300k D300" in report
