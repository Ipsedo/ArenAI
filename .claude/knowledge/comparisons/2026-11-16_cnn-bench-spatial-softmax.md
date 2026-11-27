# CNN test bench: spatial softmax does not generalise lateral localisation

Runs: 522 (save 68 images), 470 (cited)

2026-11-16: supervised bench on images of 522 s68 — spatial softmax generalises nothing (left/right 0.51) even with the exact label; flattened outputs learn and rise with data (7 convs flattened 1×2: 0.68, 0.73 < 100 m). Follow-up of the alignment-decoding note of 522 save_68 (2026-11-16); Samuel's hypothesis « le spatial softmax n'est pas adapté », ready to revisit the CNN entirely. Report: outputs/train_522_ppo_liquid/frames/bench.md (scripts bench_cnn.py, bench_curve.py; images save_68_side550_40ep.bin, 3.6 GB).

- 40 episodes, 13.6 k images with an enemy in the field; target = yaw/elevation of the most on-axis enemy; test on 8 never-seen episodes.
- A (current, 4 convs + spatial softmax): left/right 0.51-0.52 at 8/16/32 episodes, corr 0; 16×32 map and coordinates-as-input variants same.
- B (4 convs flattened 8×16) 0.62; C (6 convs flattened 2×4) 0.66; D (7 convs flattened 1×2, form before 2026-10-20) 0.57 → 0.61 → 0.68, target < 100 m 0.73, median error 2.4° (3.7° without image). Curves still rising.
- Single-enemy images no easier ⟹ not ambiguity. Hard task: random colours per episode, small tanks, agent tank at bottom-centre (3rd-person camera) masking close aligned targets.
- All memorise training; generalisation = the criterion.

**Why:** first direct (non-RL) proof that the spatial softmax introduced 2026-10-20 (dbd11417) prevents lateral localisation; consistent with the undecodable yaw of 470 and 522.

**Use:** run candidate = back to a flattened output with 6-7 stride-2 convs (Samuel's decision, architecture change). Limits: 10.8 k images, one seed, supervised ≠ RL. Tool: `arenai_agent_probe --frames <bin> --frames_every N --zoom off` (header 10 floats: episode, step, tank, view_distance/yaw/pitch, best_distance/yaw/pitch, view_count; then 3×128×256 uint8); `AimGeometry` gained best_yaw, best_pitch, view_count (uncommitted at 2026-11-16).

**Architecture choice (2026-11-16 evening, bench_seeds.txt, 4 seeds, 32 episodes)**, left/right / target < 100 m / median yaw error < 100 m: 5 convs 8-16-32-48-64 flattened 4×8 (default of 10-20) 0.599 / 0.630 / 3.45°; 5 convs 16…128-128 flattened 4×8 0.630 / 0.693 / 3.21°; 6 convs flattened 2×4×128 0.653 / 0.713 / 2.84°; 6 convs 64 channels 0.637 / 0.710 / 2.71°; 7 convs flattened 1×2×128 0.654 / 0.714 / 2.45° (std 0.01-0.03). GPU cost identical (inference 32: 0.35-0.41 ms; training 512: 27-30 ms): cost is in the first layers. Proposed to Samuel: 7 convs 16-32-64-128-128-128-128, GroupNorm 2-4-8-16-16-16-16, Flatten → 256 (liquid input 320 instead of 448). Open point: spatial-softmax checkpoints (incl. the committed game model) no longer load if the layer is replaced without an option.
(see docs/decisions.md 2026-11-25: spatial softmax removed)
