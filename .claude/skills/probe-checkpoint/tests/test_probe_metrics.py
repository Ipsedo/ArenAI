import csv
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import probe_metrics as pm  # noqa: E402

DECISION = {
    "episode": 0, "step": 10, "tank": 0, "distance": 50, "best_angle": 90, "best_distance": 50,
    "can_fire": 1, "shells": 1, "p_fire": 0, "p_zoom": 0, "fire": 0,
    "mu_dir_x": 0, "mu_dir_y": 0, "mu_canon_x": 0, "mu_canon_y": 0,
    "sigma_canon_x": 0.1, "sigma_canon_y": 0.1,
}
SHOT = {
    "episode": 0, "tank": 0, "fire_best_angle": 90, "fire_best_distance": 50,
    "group_size": 1, "group_hits": 0, "group_miss": 0,
}


def write_csv(path, defaults, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(defaults), delimiter=";")
        writer.writeheader()
        for r in rows:
            writer.writerow({**defaults, **r})


def analyze(tmp_path, decisions, shots=None, step_seconds=0.1):
    path = tmp_path / "save_1_side550.csv"
    write_csv(path, DECISION, decisions)
    if shots is not None:
        write_csv(tmp_path / "save_1_side550_shots.csv", SHOT, shots)
    return pm.analyze(str(path), step_seconds)


def is_nan(x):
    return isinstance(x, float) and math.isnan(x)


# helpers

def test_mean_ignores_nan_and_is_nan_when_empty():
    assert pm.mean([1.0, float("nan"), 3.0]) == 2.0
    assert is_nan(pm.mean([]))
    assert is_nan(pm.mean([float("nan")]))


def test_ratio_is_nan_on_zero_or_nan_operand():
    assert pm.ratio(1.0, 4.0) == 0.25
    assert is_nan(pm.ratio(1.0, 0.0))
    assert is_nan(pm.ratio(float("nan"), 2.0))
    assert is_nan(pm.ratio(1.0, float("nan")))


def test_quantile_on_eleven_values():
    xs = [10, 0, 9, 1, 8, 2, 7, 3, 6, 4, 5]
    assert pm.quantile(xs, 0.1) == 1
    assert pm.quantile(xs, 0.5) == 5
    assert pm.quantile(xs, 0.9) == 9
    assert is_nan(pm.quantile([], 0.5))


def test_load_skips_rows_that_are_not_numeric(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("step;shells\n1;0.5\nx;0.5\n3\n4;0.25\n")
    assert pm.load(str(path)) == [{"step": 1.0, "shells": 0.5}, {"step": 4.0, "shells": 0.25}]


def test_tanh_normal_entropy_tends_to_gaussian_entropy_for_small_sigma():
    sigma = 1e-4
    gaussian = 0.5 * math.log(2 * math.pi * math.e * sigma * sigma)
    assert pm.tanh_normal_entropy(0.0, sigma) == pytest.approx(gaussian, abs=1e-6)


def test_tanh_normal_entropy_drops_with_saturation():
    assert pm.tanh_normal_entropy(3.0, 0.3) < pm.tanh_normal_entropy(0.0, 0.3)


def test_yaw_spread_is_zero_without_noise():
    r = {"mu_yaw_cos": 0.4, "mu_yaw_sin": -0.7, "sigma_yaw_cos": 0.0, "sigma_yaw_sin": 0.0}
    assert pm.yaw_spread_deg(r) == 0.0


# armed population

def test_armed_is_can_fire_and_reserve_and_step_at_least_10(tmp_path):
    m = analyze(tmp_path, [
        {"step": 9, "p_fire": 1.0},
        {"step": 10, "p_fire": 0.2},
        {"step": 11, "can_fire": 0, "p_fire": 1.0},
        {"step": 12, "can_fire": 0.5, "p_fire": 1.0},
        {"step": 13, "shells": 0, "p_fire": 1.0},
    ])
    assert m["decisions"] == 5
    assert m["armed_fraction"] == pytest.approx(1 / 4)
    assert m["trigger"]["p_fire_armed"] == pytest.approx(0.2)


def test_no_armed_decision_gives_nan_trigger(tmp_path):
    m = analyze(tmp_path, [{"step": s, "shells": 0, "p_fire": 0.9} for s in (10, 11, 12)])
    assert m["armed_fraction"] == 0.0
    t = m["trigger"]
    assert is_nan(t["p_fire_armed"]) and is_nan(t["p_fire_aligned_close"])
    assert is_nan(t["p_fire_off_axis"]) and is_nan(t["ratio"]) and is_nan(t["gap"])
    assert t["n_aligned"] == t["n_aligned_close"] == t["n_off_axis"] == 0
    assert "corr_p_fire_p_zoom" not in t


# trigger selectivity

TRIGGER_ROWS = [
    {"best_angle": 4.99, "best_distance": 39.9, "p_fire": 0.8},
    {"best_angle": 4.0, "best_distance": 40.0, "p_fire": 0.4},
    {"best_angle": 3.0, "best_distance": -1.0, "p_fire": 0.2},
    {"best_angle": 5.0, "best_distance": 10.0, "p_fire": 0.6},
    {"best_angle": 19.99, "best_distance": 10.0, "p_fire": 0.5},
    {"best_angle": 20.0, "best_distance": 10.0, "p_fire": 0.1},
    {"best_angle": 90.0, "best_distance": 10.0, "p_fire": 0.3},
]


def test_trigger_selectivity_populations_and_ratio(tmp_path):
    t = analyze(tmp_path, TRIGGER_ROWS)["trigger"]
    assert (t["n_aligned"], t["n_aligned_close"], t["n_off_axis"]) == (3, 1, 2)
    assert t["p_fire_aligned_close"] == pytest.approx(0.8)
    assert t["p_fire_aligned"] == pytest.approx((0.8 + 0.4 + 0.2) / 3)
    assert t["p_fire_off_axis"] == pytest.approx(0.2)
    assert t["p_fire_armed"] == pytest.approx(2.9 / 7)
    assert t["ratio"] == pytest.approx(4.0)
    assert t["ratio_angle_only"] == pytest.approx((1.4 / 3) / 0.2)
    assert t["gap"] == pytest.approx(0.6)


def test_trigger_ignores_spawn_steps_and_disarmed_decisions(tmp_path):
    noise = [
        {"step": 9, "best_angle": 1.0, "best_distance": 10.0, "p_fire": 0.0},
        {"shells": 0, "best_angle": 1.0, "best_distance": 10.0, "p_fire": 0.0},
        {"can_fire": 0, "best_angle": 60.0, "p_fire": 1.0},
    ]
    t = analyze(tmp_path, TRIGGER_ROWS + noise)["trigger"]
    assert (t["n_aligned"], t["n_aligned_close"], t["n_off_axis"]) == (3, 1, 2)
    assert t["ratio"] == pytest.approx(4.0)


def test_deterministic_trigger_is_share_of_p_fire_above_half(tmp_path):
    t = analyze(tmp_path, TRIGGER_ROWS + [
        {"best_angle": 1.0, "best_distance": 10.0, "p_fire": 0.5},
        {"best_angle": 30.0, "p_fire": 0.51},
    ])["trigger"]
    assert t["deterministic_fires_aligned"] == pytest.approx(1 / 4)
    assert t["deterministic_fires_off_axis"] == pytest.approx(1 / 3)


def test_trigger_by_angle_buckets_are_half_open(tmp_path):
    by = analyze(tmp_path, TRIGGER_ROWS)["trigger"]["by_angle"]
    assert list(by) == ["0-2", "2-5", "5-10", "10-20", "20-45", "45-180"]
    assert [by[k]["n"] for k in by] == [0, 3, 1, 1, 1, 1]
    assert is_nan(by["0-2"]["p_fire"])
    assert by["2-5"]["p_fire"] == pytest.approx((0.8 + 0.4 + 0.2) / 3)
    assert by["5-10"]["p_fire"] == pytest.approx(0.6)
    assert by["10-20"]["p_fire"] == pytest.approx(0.5)
    assert by["20-45"]["p_fire"] == pytest.approx(0.1)
    assert by["45-180"]["p_fire"] == pytest.approx(0.3)


def test_ratio_is_nan_without_off_axis_decision(tmp_path):
    t = analyze(tmp_path, [{"best_angle": 1.0, "best_distance": 10.0, "p_fire": 0.8}])["trigger"]
    assert t["p_fire_aligned_close"] == pytest.approx(0.8)
    assert is_nan(t["ratio"]) and is_nan(t["ratio_angle_only"])


def test_ratio_is_nan_when_off_axis_trigger_is_zero(tmp_path):
    t = analyze(tmp_path, [
        {"best_angle": 1.0, "best_distance": 10.0, "p_fire": 0.8},
        {"best_angle": 60.0, "p_fire": 0.0},
    ])["trigger"]
    assert is_nan(t["ratio"])
    assert t["gap"] == pytest.approx(0.8)


def test_corr_p_fire_p_zoom_is_pearson_over_armed(tmp_path):
    rows = [{"p_fire": p, "p_zoom": z} for p, z in ((0.1, 0.3), (0.2, 0.2), (0.3, 0.1))]
    rows.append({"shells": 0, "p_fire": 0.9, "p_zoom": 0.9})
    assert analyze(tmp_path, rows)["trigger"]["corr_p_fire_p_zoom"] == pytest.approx(-1.0)


def test_corr_p_fire_p_zoom_is_nan_on_constant_input(tmp_path):
    rows = [{"p_fire": p, "p_zoom": 0.5} for p in (0.1, 0.2, 0.3)]
    assert is_nan(analyze(tmp_path, rows)["trigger"]["corr_p_fire_p_zoom"])


def test_corr_p_fire_p_zoom_needs_three_armed_decisions(tmp_path):
    rows = [{"p_fire": 0.1, "p_zoom": 0.3}, {"p_fire": 0.2, "p_zoom": 0.2}]
    assert "corr_p_fire_p_zoom" not in analyze(tmp_path, rows)["trigger"]


# turret

def test_turret_shares_cover_every_decision_from_step_10(tmp_path):
    tu = analyze(tmp_path, [
        {"step": 9, "best_angle": 0.0, "best_distance": 1.0, "distance": 1.0},
        {"best_angle": 4.99, "best_distance": 30.0, "distance": 20.0},
        {"best_angle": 5.0, "best_distance": 60.0, "distance": 40.0, "shells": 0},
        {"best_angle": 9.99, "best_distance": 90.0, "distance": 60.0, "can_fire": 0},
        {"best_angle": 19.99, "best_distance": -1.0, "distance": 80.0},
        {"best_angle": 20.0, "best_distance": -1.0, "distance": 100.0},
    ])["turret"]
    assert tu["below_5deg"] == pytest.approx(1 / 5)
    assert tu["below_10deg"] == pytest.approx(3 / 5)
    assert tu["below_20deg"] == pytest.approx(4 / 5)
    assert tu["median_best_angle"] == 9.99
    assert tu["median_best_distance"] == 60.0
    assert tu["median_nearest_distance"] == 60.0


# policy heads

def test_sigma_median_and_ceiling_share(tmp_path):
    h = analyze(tmp_path, [
        {"sigma_canon_x": 0.1, "sigma_canon_y": 0.2},
        {"sigma_canon_x": 0.69, "sigma_canon_y": 0.3},
        {"sigma_canon_x": 0.7, "sigma_canon_y": 0.68},
        {"step": 9, "sigma_canon_x": 0.9, "sigma_canon_y": 0.9},
    ])["heads"]
    assert h["aim_dims"] == ["canon_x", "canon_y"]
    assert h["sigma_canon_x"]["p50"] == 0.69
    assert h["sigma_canon_x"]["at_ceiling"] == pytest.approx(2 / 3)
    assert h["sigma_canon_y"]["p50"] == 0.3
    assert h["sigma_canon_y"]["at_ceiling"] == 0.0


def test_mu_median_and_saturation_share(tmp_path):
    h = analyze(tmp_path, [
        {"mu_canon_x": -1.6, "mu_dir_x": 0.1},
        {"mu_canon_x": 1.5, "mu_dir_x": 0.2},
        {"mu_canon_x": 0.4, "mu_dir_x": 0.3},
        {"step": 9, "mu_canon_x": 5.0, "mu_dir_x": 5.0},
    ])["heads"]
    assert h["all_dims"] == ["dir_x", "dir_y", "canon_x", "canon_y"]
    assert h["mu_canon_x"]["mean"] == pytest.approx(0.1)
    assert h["mu_canon_x"]["abs_p50"] == 1.5
    assert h["mu_canon_x"]["saturated"] == pytest.approx(1 / 3)
    assert h["mu_dir_x"]["saturated"] == 0.0
    assert h["mu_dir_x"]["mean"] == pytest.approx(0.2)


def test_p_zoom_is_mean_over_decisions_from_step_10(tmp_path):
    h = analyze(tmp_path, [
        {"step": 9, "p_zoom": 1.0}, {"p_zoom": 0.2}, {"p_zoom": 0.6, "shells": 0},
    ])["heads"]
    assert h["p_zoom"] == pytest.approx(0.4)
    assert "yaw_spread_deg" not in h


# reserve

def reserve_rows(empty_steps_by_tank_episode, nb_steps=12):
    rows = []
    for (episode, tank), empty in empty_steps_by_tank_episode.items():
        for step in reversed(range(nb_steps)):
            rows.append({"episode": episode, "tank": tank, "step": step, "can_fire": 0,
                         "shells": 0 if step in empty else 0.5})
    return rows


def test_reserve_first_empty_emptied_share_and_disarmed_share(tmp_path):
    rows = reserve_rows({(0, 0): {3, 4}, (0, 1): set(), (1, 0): {1, 11}, (1, 1): {2}})
    m = analyze(tmp_path, rows, step_seconds=0.5)
    r = m["reserve"]
    assert (m["episodes"], m["tanks"], m["decisions"]) == (2, 2, 48)
    assert r["tank_episodes"] == 4
    assert r["emptied"] == pytest.approx(3 / 4)
    assert r["first_empty_s_p50"] == pytest.approx(1.0)
    assert r["disarmed_fraction"] == pytest.approx(5 / 48)
    assert r["life_s_p50"] == pytest.approx(6.0)


def test_reserve_never_emptied(tmp_path):
    r = analyze(tmp_path, reserve_rows({(0, 0): set(), (0, 1): set()}))["reserve"]
    assert r["emptied"] == 0.0
    assert is_nan(r["first_empty_s_p50"])
    assert is_nan(r["first_empty_s_p10"]) and is_nan(r["first_empty_s_p90"])
    assert r["disarmed_fraction"] == 0.0


def test_reserve_first_empty_is_the_earliest_empty_step_of_the_tank_episode(tmp_path):
    r = analyze(tmp_path, reserve_rows({(0, 0): {7, 2, 9}}), step_seconds=0.1)["reserve"]
    assert r["emptied"] == 1.0
    assert r["first_empty_s_p50"] == pytest.approx(0.2)
    assert r["disarmed_fraction"] == pytest.approx(3 / 12)


# shots

def test_fired_counts_fire_with_can_fire_on_every_step(tmp_path):
    s = analyze(tmp_path, [
        {"step": 0, "fire": 1},
        {"step": 10, "fire": 1},
        {"step": 11, "fire": 1, "can_fire": 0},
        {"step": 12, "fire": 0},
        {"step": 13, "fire": 0.5},
    ], step_seconds=0.2)["shots"]
    assert s["fired"] == 2
    assert s["per_tank_second"] == pytest.approx(2 / (5 * 0.2))
    assert "hit_per_shell" not in s and "by_angle" not in s


SHOT_ROWS = [
    {"fire_best_angle": 4.99, "fire_best_distance": 39.0, "group_hits": 1, "group_miss": 0.0},
    {"fire_best_angle": 0.0, "fire_best_distance": 40.0, "group_hits": 1, "group_miss": 2.0},
    {"fire_best_angle": 5.0, "fire_best_distance": 10.0, "group_hits": 0, "group_miss": 10.0},
    {"fire_best_angle": 19.9, "fire_best_distance": 10.0, "group_hits": 1, "group_miss": 4.0},
    {"fire_best_angle": 20.0, "fire_best_distance": 10.0, "group_hits": 0, "group_miss": 30.0},
]
SHOT_DECISIONS = [
    {"best_angle": 1.0}, {"best_angle": 30.0}, {"best_angle": 40.0}, {"best_angle": 50.0},
    {"best_angle": 1.0, "shells": 0},
]


def test_hit_per_shell_and_shot_buckets_with_single_shell_groups(tmp_path):
    s = analyze(tmp_path, SHOT_DECISIONS, SHOT_ROWS)["shots"]
    assert s["landed"] == 5
    assert s["hit_per_shell"] == pytest.approx(3 / 5)
    assert s["miss_m_p50"] == 4.0
    by = s["by_angle"]
    assert list(by) == ["0-5", "5-20", "20-180"]
    assert by["0-5"]["share"] == pytest.approx(2 / 5)
    assert by["0-5"]["hit_per_shell"] == pytest.approx(1.0)
    assert by["5-20"]["share"] == pytest.approx(2 / 5)
    assert by["5-20"]["hit_per_shell"] == pytest.approx(1 / 2)
    assert by["20-180"]["share"] == pytest.approx(1 / 5)
    assert by["20-180"]["hit_per_shell"] == 0.0
    assert s["aligned_close_share"] == pytest.approx(1 / 5)


def test_enrichment_is_aligned_shot_share_over_aligned_armed_decision_share(tmp_path):
    s = analyze(tmp_path, SHOT_DECISIONS, SHOT_ROWS)["shots"]
    assert s["enrichment"] == pytest.approx((2 / 5) / (1 / 4))


def test_enrichment_is_nan_without_aligned_armed_decision(tmp_path):
    s = analyze(tmp_path, [{"best_angle": 30.0}], SHOT_ROWS)["shots"]
    assert is_nan(s["enrichment"])


def test_empty_shot_bucket_has_zero_share_and_nan_hit_rate(tmp_path):
    s = analyze(tmp_path, SHOT_DECISIONS, SHOT_ROWS[2:])["shots"]
    assert s["by_angle"]["0-5"]["share"] == 0.0
    assert is_nan(s["by_angle"]["0-5"]["hit_per_shell"])
    assert s["aligned_close_share"] == 0.0


def test_shots_file_without_rows_is_read_as_no_shot(tmp_path):
    s = analyze(tmp_path, SHOT_DECISIONS, [])["shots"]
    assert "hit_per_shell" not in s


# headline

def test_headline_labels_and_sources(tmp_path):
    decisions = TRIGGER_ROWS + reserve_rows({(1, 0): {3}, (1, 1): set(), (1, 2): {5}})
    m = analyze(tmp_path, decisions, SHOT_ROWS, step_seconds=0.5)
    values = {label: read(m) for label, read in pm.HEADLINE}
    assert list(values) == [
        "ratio", "p_al", "p_off", "p_arm", "hit/ob", "h<5°", "tirs<5°", "enrich", "vide_s",
        "%vident", "%désarmé", "tir/s", "tour<5°", "σx", "σy", "σ@plaf", "|μ|cx", "μsat",
        "σyaw°", "p_zoom",
    ]
    t, s, r, h = m["trigger"], m["shots"], m["reserve"], m["heads"]
    assert values["ratio"] == t["ratio"] == pytest.approx(4.0)
    assert values["p_al"] == t["p_fire_aligned_close"]
    assert values["p_off"] == t["p_fire_off_axis"]
    assert values["p_arm"] == t["p_fire_armed"]
    assert values["hit/ob"] == s["hit_per_shell"] == pytest.approx(3 / 5)
    assert values["h<5°"] == s["by_angle"]["0-5"]["hit_per_shell"]
    assert values["tirs<5°"] == s["by_angle"]["0-5"]["share"]
    assert values["enrich"] == s["enrichment"]
    assert values["vide_s"] == r["first_empty_s_p50"]
    assert values["%vident"] == r["emptied"] == pytest.approx(2 / 4)
    assert values["%désarmé"] == r["disarmed_fraction"]
    assert values["tir/s"] == s["per_tank_second"]
    assert values["tour<5°"] == m["turret"]["below_5deg"]
    assert values["σx"] == h["sigma_canon_x"]["p50"]
    assert values["σy"] == h["sigma_canon_y"]["p50"]
    assert values["|μ|cx"] == h["mu_canon_x"]["abs_p50"]
    assert values["p_zoom"] == h["p_zoom"]
    assert is_nan(values["σyaw°"])


def test_headline_ceiling_and_saturation_take_the_worst_dimension(tmp_path):
    m = analyze(tmp_path, [
        {"sigma_canon_x": 0.7, "sigma_canon_y": 0.1, "mu_dir_y": 2.0, "mu_canon_x": 0.0},
        {"sigma_canon_x": 0.1, "sigma_canon_y": 0.1, "mu_dir_y": 2.0, "mu_canon_x": 2.0},
        {"sigma_canon_x": 0.1, "sigma_canon_y": 0.1, "mu_dir_y": 0.0, "mu_canon_x": 0.0},
    ])
    values = {label: read(m) for label, read in pm.HEADLINE}
    assert values["σ@plaf"] == pytest.approx(1 / 3)
    assert values["μsat"] == pytest.approx(2 / 3)


def test_headline_shot_columns_are_nan_without_shots_file(tmp_path):
    m = analyze(tmp_path, TRIGGER_ROWS)
    values = {label: read(m) for label, read in pm.HEADLINE}
    for label in ("hit/ob", "h<5°", "tirs<5°", "enrich"):
        assert is_nan(values[label])
