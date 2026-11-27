# Coloured AR(1) exploration noise — coded 2026-11-18, removed 2026-11-19

Runs: 529, 530, 531, 532, 533, 534, 536, 537, 538

2026-11-18: hypothesis retained after the failure of the critic-side pilots (530-532): per-decision white Gaussian noise (1.2°/0.1 s) produces no different events, so the event reward does not reach the turret (dwell 0.17 s, tour<5° 8.5 % vs 21-27 % for 520/522 — probes 529 s27, the gap is not a curriculum artefact). Samuel chose coloured AR(1) noise over strict gSDE, ρ 0.9 default (`--noise_correlation`), on wheels + turret. Actor batch 2400 refused (« ça tacle pas le pb »).

Code (uncommitted): `distributions/colored_noise.{h,cpp}`, shared agent + collector; marginal law, log-prob, entropy and α unchanged; reset at each trajectory start (episode end, done or truncated per tank — requested by Samuel: independent trajectories); game (factory.cpp) passes 0 since deterministic; probe.cpp reads `noise_correlation` from config (0 if absent, SAC).

**Why:** Samuel refuses Φ because it imposes a target, not on principle against a dense signal; he wants an irrefutable event reward.
Run 532 killed at 249 k (D 450, never measured at D 550). Samuel launched the next run himself in CLion. Judge on probes: dwell < 5°, tour<5°, at equal D. See comparisons/2026-11-18_critic-blind-to-aiming-gesture.md.

**Removed 2026-11-19 (Samuel's request after control 538):** class `ColoredNoise`, option `--noise_correlation`, overload `gaussian_tanh_sample(mu, sigma, noise)` and tests deleted from the 3 PPO agents, the probe and `sigma_gradients` (uncommitted, agent tests green in `build/`). Reason: white noise ≥ coloured at equal config, the policy cancels coloured noise, KL regime (runs 537, comparisons/2026-11-19_aim-gesture-state-vs-events-signal.md). Never propose again. Checkpoints 533-537 are now probed in white noise. (see docs/decisions.md 2026-11-21: removed in `0137ae77`)
