# Local optimum of aim/trigger coordination (oracle 2×2×2 plan on 520 save_37, confirmed 522 save_68)

Runs: 520 (save_37), 522 (save_68), 510-520 (common ceiling)

2026-11-15: 2×2×2 oracle plan (yaw/elevation/trigger) on 520 save_37: no single link pays, perfect aim + agent trigger = policy level, only the oracle trio ×1.8/×2.6 ⟹ local coordination optimum; the trigger ignores the alignment produced by the oracle.

Hits/tank-s (550/1000): AAA 0.320/0.140; one oracle link: yaw 0.295/0.106, elevation 0.348/0.158, trigger 0.124/0.099; OOA (oracle aim, agent trigger, turret aligned 84-92 %) 0.320/0.150; OAO 0.200/0.181; AOO 0.169/0.126; OOO 0.586/0.361. In OOA, p_fire at < 1° = 0.13/0.07 vs 0.43-0.50 aligned in normal play.

Probe: `--oracle_policy_yaw`, `--oracle_policy_fire` added (validated by Samuel), oracle trigger = yaw AND elevation < 1° (uncommitted at 2026-11-15). Report: outputs/train_520_ppo_liquid/probes/oracle/save_37.md.

Parallax + dynamics test (2026-11-15): OOA centred camera p_fire aligned 0.12 → 0.16 (agent alone 0.58) ⟹ parallax minor. The trigger fires in self-sustained bursts: p_fire 0.63 in the second after a shot then 0.1-0.2; under OOO (oracle fires) 0.45 constant, under OOA (burst never primed) 0.18 → 0.06. Fire decision = internal engagement state (recent shot, proprio cooldown/reserve or liquid memory) > visible alignment. Proposed next tests: neutralise cooldown/reserve in the proprio (probe option), trigger fine-tune under oracle aim.

**Why:** explains the common ceiling of 510-520: links are co-adapted (loose trigger ~5° + volume ↔ aim rarely < 1°), marginal gradient of each link ≈ 0. Revises comparisons/2026-11-12_oracle-aim-ceiling-elevation-parallax.md (« elevation is the main lock »): elevation alone pays nothing.

**Rule:** no longer propose tuning a single link as a lever of the ceiling. Proposed next tests: OOA + `--oracle_camera_center` (H1 parallax), then alignment decoding from features (H2). Levers to discuss with Samuel: on-axis camera, weaned assisted aim, trigger fine-tune under oracle aim. Related run 520 (discrete target 0.01).

**Confirmed on 522 save_68 (2026-11-16, outputs/train_522_ppo_liquid/probes/oracle/save_68.md):** hits/tank-s agent 0.152/0.111; oracle trigger + agent aim 0.128/0.087; oracle aim + agent trigger 0.147/0.059 (turret aligned 93-97 %, aligned p_fire < 2° 0.115 → 0.019/0.008); all oracle 0.533/0.448 (×3.5/×4). At equal angle and distance (< 2°, 50-100 m, 80 % hits) p_fire 0.20 with its own aim vs 0.06 under oracle. Aligned < 100 m: 0.23 after a hit impact, 0.16 after a miss, 0.08 without recent shot (0.016 under oracle) ⟹ the first shot of an engagement is the bottleneck, shot outcome matters little (not a mere refund sensor), |μ canon| does not explain the gap. Run 522's empty penalty moved the optimum toward precision without breaking it. Acquisition unchanged since 2026-11-12: nearest < 100 m in field 38 % of time, median bearing 76-82°. Levers proposed to Samuel 2026-11-16 (nothing decided): training under weaned assisted aim (breaks co-adaptation), preceded by a short trigger fine-tune under oracle aim as causal test; no resume-training option in main.cpp. (see docs/decisions.md 2026-11-16: training under assisted/oracle aim refused)
