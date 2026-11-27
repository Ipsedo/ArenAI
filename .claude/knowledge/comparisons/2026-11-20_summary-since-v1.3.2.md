# Summary since release v1.3.2 — runs 523 to 542

Runs: 522 (release reference), 523-542 (focus 524, 525, 526, 527-529, 530-532, 533-538, 539, 540, 541, 542).
Date: 2026-11-20.

- Reference release: v1.3.2 (0c678b42, 16/11 08:54) = state of run 522.
- Since then: 5 commits on develop (cfb76a2e → b83da585) + uncommitted tree (60 source files).
- Release marker, CSV of 522 at D 1000 over 1 640 rows (800 k → 1.9 M): kill 19.2, hit 0.38, fire 0.28/s, `r` +0.0025. No run since has held D 1000.

## 1. Each change, its problem, its effect

| # | Change (run) | Problem targeted | Measured effect | Verdict | In tree |
|---|---|---|---|---|---|
| 1 | Flattened 7-conv CNN instead of spatial softmax (other-PC witness, 523+) | softmax does not localise laterally (left/right 0.51 supervised) | left/right decodable 0.75-0.79; witness with separate credit: turret<5° 29 → 40 %, kill 17.8 at D 1000 ≡ 522 | real gain on perception, none on kills at D 1000 | yes |
| 2 | Single advantage: removal of the separate cannon credit and per-group KL stop (523, 524) | let the critic credit aim | turret<5° 16 → 8 %, kill 8 at D 1000 versus 17; cause isolated by the witness | negative, established | replaced by 13 |
| 3 | target_kl 0.05 → 0.10 (524) | the single KL stop cut 45 % of minibatches | skip still 20-34 % | neutral | yes |
| 4 | Φ removed from reward, pure event (525) | Φ aims at the enemy closest to the trajectory, bias | ≡ 524 at equal D | neutral: Φ in the reward alone did nothing | yes |
| 5 | Actor batch 512 → 2400 (526, default since 537) | per-step actor noise | kl ÷5, skip 0; aim unchanged, trigger slower | cleans the optimisation, not the skill | yes |
| 6 | Curriculum on kills per tank, 0.3 / 0.5 (527, 528) | the old hits/shots criterion was satisfied by closing the trigger | D settles where kills ≈ 16 (D 750-850); makes the drop visible, does not fix it | more honest instrument, not a gain | yes |
| 7 | 5 shells per hit instead of 2 (528) | trigger closed beyond 50 m (firing profitable only at 50 % success) | aligned p_fire 50-100 m 0.44 at s9 versus 0.14; then same closure 100-130 k later | delays, does not lift | yes |
| 8 | σ canon cap 0.25, σ init 0.20, canon entropy target −1 (529) | σ 0.5 made target holding unreachable | dwell ×2 at s5-s9 then sweep returned through μ; D never > 700 | hypothesis refuted | yes |
| 9 | Critic pilots: epochs 2 + lr 1e-4 (530), γ 0.99 (531), short value head γ 0.9 (532) | critic blind to alignment | slope of V null in all three; 531 degrades the actor | negative | no |
| 10 | Coloured AR(1) exploration noise (533-537) | corrective gesture drowned in white noise | control 538: white noise ≥ coloured everywhere | negative | no (removed) |
| 11 | Back to spatial softmax + linear command without a·\|a\| (536) | KL skip, growing μ head | skip 0.9, D stayed at 250 | softmax negative; a·\|a\| removal never isolated | a·\|a\| removed: yes |
| 12 | Up-to-date image, fix of one-decision lag (539) | the agent saw t−1 | dwell < 5° ×2 at s3-s5, kill +2 to +4 up to D 700; lost at s12 (0.24 s), kill 16 at D 800 | real bug fixed, gain not held | yes |
| 13 | Cannon credit on H, critic head, per-group KL stop (540 with 3rd CNN, 541, 542) | give the cannon a one-step signal without Φ | see section 2 | not the expected lever | yes |
| 14 | `discrete_target_proba` 0.01 → 0.05 (542) | trigger extinction | inert up to 450 k; `alpha_fire` 1e-4 → 4.5e-4 without retaining p_fire (0.063 → 0.024) | no visible effect | yes |

Overall reading: two fundamental fixes (1, 12), three hygiene changes (5, 6, 7), the rest neutral or negative. No change brought back the release's level.

## 2. Runs with H (540, 541, 542) versus 539

Probes at side 550, 4 episodes, stochastic.

| run save | kills | hits/tank-s | hit/shell | yaw < 5° at < 100 m | dwell | μ/yaw gain | corrector | setpoint °/dec |
|---|---|---|---|---|---|---|---|---|
| 539 s3 | 19.8 | 0.067 | 7.5 % | 8.0 % | 0.62 s | −0.012 | 0.59 | 1.2 |
| 539 s5 | 24.0 | 0.082 | 9.1 % | 9.1 % | 0.73 s | −0.014 | 0.59 | 1.0 |
| 539 s12 | 21.0 | 0.053 | 20.8 % | 5.5 % | 0.23 s | −0.023 | 0.51 | 3.1 |
| 541 s3 | 15.2 | 0.046 | 6.8 % | 4.2 % | 1.13 s | −0.002 | 0.51 | 0.6 |
| 541 s9 | 27.5 | 0.088 | 15.4 % | 12.4 % | 0.78 s | −0.011 | 0.58 | 1.7 |
| 541 s12 | 25.8 | 0.073 | 16.9 % | 10.7 % | 0.64 s | −0.010 | 0.53 | 2.5 |
| 541 s16 | 2.5 | 0.007 | 22.3 % | 4.1 % | 0.67 s | −0.001 | 0.51 | 0.6 |
| 541 s20 | 27.0 | 0.066 | 16.6 % | 11.3 % | 1.28 s | −0.008 | 0.57 | 1.2 |
| 541 s26 | 20.8 | 0.062 | 28.6 % | 11.0 % | 0.75 s | −0.009 | 0.56 | 2.1 |
| 542 s3 | 22.2 | 0.069 | 8.2 % | 7.7 % | 1.32 s | −0.004 | 0.57 | 0.5 |
| 542 s9 | 20.5 | 0.057 | 10.3 % | 7.3 % | 0.88 s | −0.005 | 0.55 | 0.8 |
| 542 s13 | 23.0 | 0.059 | 15.1 % | 7.6 % | 0.69 s | −0.008 | 0.53 | 1.5 |
| 542 s16 | 20.0 | 0.048 | 19.4 % | 9.6 % | 0.86 s | −0.004 | 0.54 | 1.4 |
| 542 s19 | 12.8 | 0.043 | 27.5 % | 6.3 % | 0.87 s | −0.002 | 0.55 | 0.7 |

540 (H on a 3rd CNN): stopped at 147 k, ≡ 541 at s3, nothing beyond.

### What H does not do

- **It does not learn the aiming gesture.** μ/yaw gain −0.002 to −0.011 and corrector 0.51-0.59 on all saves of both runs: never above 539 at s3 (−0.012 / 0.59), which has no aim credit. The separate credit on Φ gave turret<5° 29 → 40 % and a dwell of 1.0 s.
- **The signal it must carry is absent.** corr(one-step ΔH, yaw reduction): 0.00 to 0.03 on 541 (s12, s14, s20), 0.00 to 0.06 on 542 (s5 to s13). The measured usefulness threshold was 0.08; the offline measurement that had motivated B gave 0.10, with a dedicated memoryless CNN on a frozen policy.
- H recognises the aligned state (σ(H) 0.13-0.25 aligned versus 0.01-0.03 off-axis, AUC 0.78-0.83) but underestimates the hit ×2.5 to 4 and its one-step difference does not track rotation.

### What H does

- **It calms the turret**, on all three runs: setpoint 0.5-0.6 °/decision at s3 versus 1.2, gain divided by 3 to 6. The long dwell (0.7-1.3 s) comes from there.
- **It avoids 539's sweep** (setpoint 3.1 °/decision and dwell 0.23 s at s12). 541 and 542 stay at 0.6-2.5 °/decision and 0.64-1.28 s.
- CSV kills at D 800: 542 22.2, 541 17.5 (25 in high phase), 539 16.3, 538 18.8. One to two runs per arm.

Reading (Claude's, untested): with a ΔH uncorrelated with the gesture, the cannon receives a normalised advantage close to noise. H's effect resembles that of "taking the cannon out of the GAE": the turret develops no sweep, stays almost still and fires in passing.

### Verdict

H is not the aim lever. Its benefit is real but indirect and capped: hits per tank-second 0.05-0.09 at side 550 over the whole duration of both runs, D 1000 never reached.

## 3. The trigger that closes: what the reward balance says

Independent of H: it affects 538, 539, 541 and 542, and 522 knew it too (trough at 420-460 k, "uniform tax on firing").

Balance per step and per tank, mean of CSV rows at given D (`rest` = r − r_hit − r_rcv = empty-reserve penalties):

| D | run | kill | fire/s | r_hit | rest | penalty / r_hit | one emptying every |
|---|---|---|---|---|---|---|---|
| 800 | 522 | 18.1 | 0.27 | +0.0038 | −0.0009 | 24 % | 90 s |
| 800 | 538 | 18.8 | 0.62 | +0.0056 | −0.0017 | 30 % | 47 s |
| 800 | 539 | 16.3 | 0.30 | +0.0032 | −0.0007 | 23 % | 108 s |
| 800 | 541 | 17.5 | 0.54 | +0.0047 | −0.0019 | 39 % | 43 s |
| 800 | 542 | 22.2 | 0.90 | +0.0057 | −0.0039 | 68 % | 21 s |
| 950 | 542 | 14.2 | 0.28 | +0.0028 | −0.0012 | 42 % | 80 s |

- In 542, from 210 k to 345 k, the empty-reserve penalty eats 70 to 80 % of the hit reward and `r` is negative. The only available response is to fire less, and that is what follows.
- Per-shell count at D 800 (hit ≈ +0.9, emptying −0.8, 10 shells + 5 per hit): a shell is profitable from 6 % success, 2.4 % at D 250. Off-axis shots (0.4-5 %) become loss-making as D rises; aligned shots (38-47 %) stay very profitable (+0.35 to +0.42 per shell).
- Yet the aligned trigger closes with the rest (542: 0.84 → 0.17; 541: 0.84 → 0.08). The aligned / off-axis logit gap stays 3.4-4.1: the actor's detector has fixed quality, only the common level moves.
- With a fixed-quality detector, the equilibrium level falls when good opportunities become rare (enemy at < 50 m: 40 % of the time at side 550, 9-18 % at side 1000) and when the false positive costs more (penalty ∝ D). The drops all start at D 750-850.
- The descent overshoots the equilibrium: at p_fire 0.02 there are almost no shots in the batch to learn that firing aligned pays, and the entropy regulator (α ≤ 5e-4) weighs nothing.

Established: the table's figures, the constant logit gap, the coincidence with D.
Inferred, not tested by intervention: that indexing the penalty on D is the cause (later ruled out: see `2026-11-20_trigger-closure-diagnosis_508-543.md` §3.5).

## 4. Open choices (as of 2026-11-20)

1. Keep or remove the credit on H. Deciding control: cannon with no credit at all (zero advantage, entropy only), judged at s9-s12 against 541/542.
2. Constant empty-reserve penalty instead of −D/1000.
3. Review the economy as a whole.
4. The only setup that held D 1000 (kill 19, hit 0.38 over 1 M steps) remains the release's: separate cannon credit on Φ, set aside on 2026-11-19 because Φ biases target choice.
