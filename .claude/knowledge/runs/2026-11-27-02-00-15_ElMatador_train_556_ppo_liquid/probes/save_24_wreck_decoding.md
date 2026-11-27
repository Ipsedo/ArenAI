# 556 save_24 — is « wreck vs living tank » in the image? (2026-11-27)

Hypothesis under test: the detached parts of a wreck make it recognisable, and the agent merely fails to use
it. Prediction if true: wreck vs living decodable from the pixels well above chance, at least at short range.
If false: near chance while a known-answer control is decoded.

Probe side 1000, 8 episodes, 32 tanks, deterministic, binary = `62ba85c7`, features + vision frames at every
decision (`outputs/train_556_ppo_liquid/features/save_24_side1000_det.*`, frames deleted, 22 GB;
script `decode_wrecks.py` next to them). Classes: living enemy within 5° of the gun axis and no wreck within
10°, vs wreck within 5° and no living enemy within 10°. Train episodes 0-5, test 6-7, balanced accuracy
(chance 0.50). Instruments: linear probe on the liquid trunk output (128) and on the encoded vision (256);
CNN trained from scratch on the 64×128 centre crop of the vision frame (3 conv blocks, 30 epochs, 2 seeds).

| target distance | trunk h | vision features | pixels CNN (2 seeds) | shuffled labels (vision / CNN) | control: tank on axis vs nothing (vision / CNN) | n test living / wreck |
|---|---|---|---|---|---|---|
| < 60 m | 0.67 | 0.66 | 0.59 / 0.60 | 0.52 / 0.52 | 0.91 / 0.95 | 679 / 826 |
| 60-100 m | 0.66 | 0.58 | 0.60 / 0.60 | 0.49 / 0.50 | 0.87 / 0.98 | 558 / 594 |
| 100-200 m | 0.62 | 0.59 | 0.69 / 0.66 | 0.50 / 0.50 | 0.80 / 0.93 | 777 / 762 |
| 200-400 m | 0.60 | 0.56 | 0.58 / 0.58 | 0.50 / 0.48 | 0.69 / 0.80 | 860 / 453 |

Trigger on the same test samples (armed): fires on living 0.43 / 0.43 / 0.14 / 0.08, on wreck 0.30 / 0.31 /
0.09 / 0.04.

## Verdict

Refuted: the image does not carry the distinction. A CNN trained for that single task on the pixels reaches
0.58-0.69, at every distance including under 60 m, while the same CNN separates « a tank on the axis » from
« nothing » at 0.80-0.98. The agent's own vision features are at the same level (0.56-0.66), and the trunk adds
little (0.60-0.67, memory included). The weak discrimination of the trigger matches what is decodable.

Limits: one checkpoint's state distribution; small CNN and 4-6 k time-correlated training samples per band, so
0.6-0.7 is this instrument's reading, not a bound on the information; the centre crop ignores the zoom state.
