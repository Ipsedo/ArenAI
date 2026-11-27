# SoftClamp — soft σ bound replaces the hard clamp (run 503 first user)

Runs: 500, 501, 502, 503 (first with SoftClamp), 504

2026-11-11 — hard σ ceiling (torch::clamp, zero gradient, 80-98 % of states at σ = 0.700) replaced by SoftClamp = lower + (upper − lower)·sigmoid(z) in misc.h/.cpp, wired on the σ head of both actors (actor.cpp, liquid_actor.cpp) with inverse init (SoftClamp::inverse) to start from initial_sigma; same bounds LOG_SIGMA_MIN/MAX; action noise σ 0.7 ≈ ±11°/decision vs tolerance 3°; **503 already runs with SoftClamp** (binary linked 02:40:58 after sources 02:36, actor_repr.txt confirms); saves ≤ 502 probed with the new binary give a wrong σ.

Option 1 of the 2026-11-11 discussion (peak-does-not-hold note, comparisons/2026-11-11_peak-does-not-hold-critic-leads-actor-follows.md), requested by Samuel: « fais le un dans une nouvelle classe SoftClamp ». Uncommitted.

**Code:** `SoftClamp(lower, upper)` in networks/misc.h/.cpp (same scheme as ConcentrationOutput), forward = lower + (upper − lower)·sigmoid(x), `SoftClamp::inverse(value, lower, upper)` = logit bounded at 1e-4; `init_log_sigma_output_weights` sets the bias to inverse(ln σ0, LOG_SIGMA_MIN, LOG_SIGMA_MAX); σ head = Linear → SoftClamp → Exp in Actor and LiquidActor. Tests: SoftClampModuleParamTest (strict bounds, 0 → middle, gradient > 0 everywhere, round-trip inverse). Class Clamp stays (tests).

**Why:** `torch::clamp` has zero gradient outside bounds ⟹ σ at the ceiling is an absorbing state, α regulates only μ. Moreover σ_max 0.7 before tanh ≈ action std 0.6 ≈ ±11° per decision (turret π rad/s = 18°/decision at 10 Hz) for a tolerance of 3° + 2.5 m: saturating μ was the only remaining route to precision (at μ = 1.5 action std falls to 0.13). SoftClamp makes descent possible, does not change the incentive to rise nor the ceiling value (option 2 = lower σ_max, option 4 = Beta law, not done).

**Pitfall:** Clamp and SoftClamp have no parameters ⟹ old checkpoints load, but the pre-activation is interpreted differently: **do not probe saves ≤ 502 with a SoftClamp binary** (503 has it ⟹ probe build/ valid; check `actor_repr.txt` of a save: Clamp vs SoftClamp) (σ wrong), like convert_mu_linear (run 501 second verdict, μ at bounds). To read on 503 (first run with SoftClamp, confounded with batch 9600 + lr ×5): fraction of states at σ ≈ 0.7 at probe (former 80-98 %), median canon σ, and whether |μ| still drifts to the bounds. See σ ceiling 0.7 (run 501), run 503 (full-rollout batch, lr 5e-4). (see docs/decisions.md 2026-11-08: state-dependent σ soft-clamped since 11-11)
