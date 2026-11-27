# Critic blind to the aiming gesture (528/529) and pilots 530-532

Runs: 528, 529, 530, 531, 532

2026-11-18, 528/529: run V does not move when the turret aligns (slope ÷10 to ÷50 vs geometric value), response spread over 4 s; critic CNN not at fault as measured first (AUC ≡ actor — later corrected, see below); event signal present (ρ 0.08-0.12 at λ ≤ 0.3 on 528 s9). Report: `outputs/comparisons/2026-11-18_leviers_avantage_unique/rapport.md` (scripts `credit.py`, `total_signal.py`, `vsens.py`, `vkernel.py`, `decode.py`, `gamma_share.py`).

Measured facts (528 s1-s14, 529 s9/s27, frozen policies):
- Run V per 10° of yaw approach (armed, no shot): 0.000-0.04; value fitted on geometry: 0.09-0.30. ρ(run advantage, gesture) ≈ 0 at λ 0, 0.005-0.02 at λ 0.95; exploitable share of advantage per continuous head ≤ 0.013.
- With the geometric value as critic: ρ yaw 0.076, elevation 0.116 at λ 0.3 on 528 s9 (Φ level), 0.02-0.03 at λ 0.95; 0.02-0.04 when the trigger is closed (529 s9/s27).
- Critic CNN: AUC aligned < 5° vs 10-39° 0.75-0.79, ≡ actor (untrained 0.58-0.67).
- V responds to present alignment with weight 0.12 (reference 0.9-1.1), the rest spread uniformly over 4 s.
- Critic target: angles = 6-12 % of return variance at γ 0.998, 25 % at γ 0.95. Critic liquid-cell input weights ×5 those of the actor, critic τ slowing (p90 1.9 → 2.7 s), actor at init.
- CSV `ev` (0.75-0.84) is a fit on training rows against a target containing V; out-of-sample corr(V, MC return) = 0.26.
- Measurement trick: config copied with gamma 0 and gae_lambda 0 ⟹ `advantage` column of sigma_gradients = −V up to a per-rollout affine.

Not settled: cause of the slow response (liquid cell, γ horizon, critic optimisation regime 8 epochs × lr 3e-4). Critic of 520/522 not measured.

**Rule:** (see docs/decisions.md 2026-11-18: asymmetric critic refused) look for levers on the critic learning side from the image. In a script, never `pgrep -f`/`pkill -f` the name of a script present in the current shell's command line (an hour lost 2026-11-18).

**Pilot 530 (2026-11-18, 529 + critic_epochs 2 + critic_learning_rate 1e-4, only difference verified by config diff):** at save_3 and save_5 V slope stays zero (s5: −0.001 to +0.007 per 10°; reference 0.11-0.13), share at delays 0-0.1 s 16 % then 12 % (529: 31 / 25 %), while critic input weights stay small (0.19 vs 1.12) and τ at init. ⟹ critic over-training is not the cause; remaining: target horizon (γ) and liquid cell. CSV ≡ 529 at 143 k.

**Pilot 531 (2026-11-18, 529 + gamma 0.99, only difference):** V slope zero at s3 and s5 (−0.005 to +0.004 per 10°; reference 0.07-0.14), weight on present alignment 0.01 then 0.07 (529: 0.18 / 0.12); degraded actor (kill 1.8 at 100 k vs 18, D stayed 250 at 140 k). ⟹ γ horizon is not the cause either; do not re-propose γ 0.99. Remaining: image → V path (critic liquid cell), untested, architecture choice for Samuel to validate.

**Offline measurement (2026-11-18, `vhead.py` / `vdistill.py`, 24 episodes of 529 s9, sigma_gradients `--critic_inputs` export uncommitted):** a memoryless value head retrained on the frozen output of the critic CNN + proprio is as blind as the run's critic (slope −0.008 to +0.006 per 10°, reference 0.15) ⟹ the liquid cell is NOT the lock. The critic CNN output barely contains alignment: AUC aligned < 5° vs 5-40° 0.66 (proprio alone 0.57), reference distillation R² 0.12. The « critic CNN not at fault » above is false: it only showed critic ≡ actor. Lock = what the CNN encodes; untested: CNN trained end to end on an alignment-dependent target.

**End-to-end CNN offline (2026-11-18, `vcnn.py`, 272 k images of 529 s9, sigma_gradients `--frames` export uncommitted):** same 7-block CNN + memoryless head. Target = geometric reference ⟹ slope +0.05 to +0.07 per 10° from the 1st pass (reference 0.11-0.13), AUC alignment 0.75, limited by overfitting. Target = Monte-Carlo return ⟹ slope ≈ 0, the network memorises noise (loss 0.13 fit / 0.95 test). ⟹ neither CNN architecture nor cell: it is the signal-to-noise ratio of alignment in the critic target (6-12 % of return variance). Look for levers there, without auxiliary head, asymmetric critic or Φ (refused).

**Short-horizon value head offline (2026-11-18, validated by Samuel as a lead: « j'aime bien »):** same protocol, one head per γ on the same CNN. Alignment AUC read in the CNN: γ 0.998 0.67; 0.99 0.72; 0.95 0.77; 0.9 0.78 (geometric target 0.75). Short-head slope +0.05 per 10° (a quarter of its reference). Double head 0.998 + 0.9/0.95: the long head goes from ~0 to +0.02-0.03 (reference 0.09-0.11). Positive but partial; one seed; partly contradicted by 531 (γ 0.99 online no effect). Not yet coded in training.

**Pilot 532 (2026-11-18, launched ~16:50):** 529 + `--short_gamma 0.9` (only difference). Uncommitted code: second output `to_short_value` of LiquidCritic (same cell), TD(λ) target at short γ with its own ReturnScaler (`short_return_scaler.pt`), summed losses, advantage unchanged (γ 0.998 head); option disabled by default (0), test `ShortValueHeadTrainsAndReloads`. Binary `build/arenai_agent/arenai_agent_train`. Criterion: V slope (long head) at saves 3 and 5 vs 529 (`data/pilote_532_s*.txt`), then CSV at equal D.

**2026-11-18 evening, Samuel's request:** short head enabled by default (`short_gamma` 0.9) for ppo, ppo_liquid and ppg, before the 532 verdict. PPO/PPG: `ValueFunction::value` returns [B, 2] (column 0 = long γ), one loss per column summed, advantage and `ev`/`v` on column 0, PPG distillation on column 0. `sigma_gradients` sets short_gamma 0 for runs whose config lacks the field (≤ 531). Uncommitted, 2443 tests OK.

**2026-11-18 ~17:50, Samuel's decision:** permanent short head (no disable option), no compatibility with old checkpoints (« pas besoin d'être compatible »), SOLID / single responsibility, anticipate without waiting for 532's proof. Refactor: class `ValueTargets` (`agents/value_targets.{h,cpp}`, module saved as `return_scaler.pt`) = λ targets per head with own γ and normaliser; critics with `nb_values` outputs (single `to_value`); the 3 trainers delegate; `--short_gamma` 0.9 default. `build/` probes no longer load runs ≤ 532: 532 probe binaries kept in `outputs/train_532_ppo_liquid/probe_bin` (collect.sh picks them if the folder exists). Pre-existing unstable test: `TruncatedNormalEdgeTest.LogPdfAndPdfConsistent` (~10 % failures).

**Pilot 532 (short head γ 0.9), save_3/5, 2026-11-18 evening:** negative at this stage. Long head +0.013 to +0.015/10° (529: −0.001 to +0.026), short head itself +0.003 to +0.016 (γ 0.9 reference: +0.15), delay 0 = 0.00; critic CNN → aligned AUC 0.55 (529 s5: 0.60-0.63). CSV: trough 40-70 k then ≡ 529 at 138 k but D 250 still at 142 k. Gap with the offline test (+0.05) unexplained: offline = D 550 policy, re-initialised CNN, MC target. Details: rapport.md §10 of folder 2026-11-18_leviers_avantage_unique. `bc` absent on the machine.
