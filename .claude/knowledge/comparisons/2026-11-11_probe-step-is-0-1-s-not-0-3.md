# Probe step = one decision = 3 frames at 30 Hz = 0.1 s (old script converted ×0.3)

Runs: 502 and earlier probe-based notes

2026-11-11 — one probe step = one decision = 3 frames at 30 Hz = 0.1 s; the script probe_an.py converted ×0.3 ⟹ every « empty reserve » in seconds written in memory before 2026-11-11 is ×3 too large (65 s = 22 s).

arenai_agent_probe: `step++` once per `env->step`, which plays action_repeat = 3 frames of wanted_frequency = 1/30 s ⟹ 0.1 s per step (1801 steps = max_episode_steps 5400 frames). The probe script reused from session to session multiplied by 0.3.

**Why it matters:** the figures « reserve 65 s / 89 s / 37.6 s » (cruise metrics at D1000 hit/shell dump note; run 502 600k verdict) remain comparable to each other but are worth ÷3.

**Rule:** convert step × 0.1 s; to compare with an old value, divide it by 3 or compare in steps.
(pitfall candidate for a skill/doc: probe step unit = 0.1 s; belongs in `probe-checkpoint` skill)
