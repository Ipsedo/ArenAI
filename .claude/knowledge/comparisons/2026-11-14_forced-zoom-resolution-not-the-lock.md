# Forced zoom: resolution is not the lock for aligned-shot precision

Runs: 510, 514 (save_23), 515 (save_9)

2026-11-14 — probe `--zoom on/off` (514 s23, 515 s9): resolution ×2 does not make aligned shots more precise < 100 m, helps only at 100-200 m; zoom transition 0.75 s in training, FOV absent from the observation.

Option `--zoom policy|on|off` added to arenai_agent_probe and probe_checkpoint.py (suffix `_zoomon`/`_zoomoff`), uncommitted.

Result: forced zoom on ⟹ hit of aligned shots by distance equal or lower than the policy (0-50 m 78-82 %, 50-100 m 46-55 %), median miss 6 m unchanged. Forced zoom off ⟹ losses only at 100-200 m (22-29 → 16-17 % for 514; 515 s9 no longer fires far, hit/shell 17 → 12 %). The earlier p_zoom ↔ hit correlation was confounded by state (the agent zooms when it is easy).

Code facts: `CanonItem::fov()` advances 2.9° per renderer call ⟹ transition 0.75 s in training/probe (1 render per decision) vs 0.25 s in game; the current FOV is not in the observation. 514 abandons zoom between 720 k and 1 M (p_zoom 0.33 → 0.09), 510 does not.

**Why:** Samuel proposed 512×256 + one CNN block as a fine-precision lever; comparisons/2026-11-12_oracle-aim-ceiling-elevation-parallax.md already said « useful resolution > 200 m only ».

**Rule:** do not present resolution as a lever of aligned-shot precision < 100 m; the lock is in the command (μ, delay, elevation), cf. comparisons/2026-11-12_aim-diagnosis-mu-white-noise.md. Analysis: outputs/comparisons/2026-11-14_zoom.md. (see docs/decisions.md 2026-11-14: 512×256 vision removed)
