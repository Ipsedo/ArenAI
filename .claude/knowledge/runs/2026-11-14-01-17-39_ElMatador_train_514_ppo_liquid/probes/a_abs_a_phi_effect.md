# 514: effect of a·|a| and centred Φ — fine aim and take-off speed

Probes 514 save_5 / 9 / 14 / 20-23 (`build/` binary with a·|a|, `--force`), compared with existing 510 and 513 probes (identical indices, same D milestones). 514 differs from 510 by a·|a|, centred Φ and economy 10/3/30; 513 differs from 510 by economy 5/30.

## 1. Trigger take-off speed (ratio, side 550 / 1000)

| save (index) | 510 | 513 | 514 |
|---|---|---|---|
| 5 (135 k) | 1.58 / 1.52 | 1.26 / 1.30 | 1.25 / 1.34 |
| 9 (243 k) | 5.9 / 10.9 | 3.1 / 3.7 | 2.9 / 4.4 |
| 14 (378 k) | 8.2 / 14.2 | 6.1 / 10.0 | 5.4 / 11.4 |

Empty reserve (s) at save_9: 510 9.5 / 14.6, 513 6.0 / 7.6, 514 8.8 / 8.7. Turret under 5° at save_14: 510 26 / 20 %, 513 26 / 22 %, 514 27 / 20 %.

514's trigger takes off at 513's pace, slower than 510. The two slower runs share one or more extra shells per hit vs 510, pointing to economy or seed, not a·|a| or Φ (which do not touch the trigger).

## 2. Fine aim of aligned shots

Shots under 5° of the aimed target, when it is also the nearest, between 50 and 200 m. Median |yaw| / |pitch| error in degrees, and share of shots that hit.

| save | 510 yaw / pitch / hit | 513 | 514 |
|---|---|---|---|
| 5 | 1.69-1.73 / 1.45-1.68 / 33-51 % (n 63-87) | 1.60-1.66 / 1.58-1.62 / 33-34 % | 1.56-1.70 / 1.50-1.61 / 37-41 % |
| 9 | 1.51-1.54 / 1.51-1.80 / 39 % | 1.60-1.88 / 1.30-1.39 / 37-38 % | 1.26-1.47 / **1.01-1.23** / **48-52 %** |
| 14 | 1.30-1.39 / 1.23-1.45 / 45-53 % | 1.34-1.36 / 1.54-1.55 / 41 % | 1.39-1.40 / 1.22-1.23 / 41 % |
| 16-18 / 20-23 | 0.90-1.47 / 0.99-1.10 / 41-52 % | 1.33-1.71 / 1.32-1.93 / 33-38 % | 1.14-1.43 / 1.04-1.73 / 41-48 % |

(n = 600-2 000 shots per cell, except 510 save_5.)

- **Pitch (canon)**: 514 reaches an error of about 1.0-1.2° from save_9, whereas 510 and 513 are then at 1.3-1.8°. 510 only gets there at save_16-28. 513 stays at 1.3-1.9° to the end. Hits at 50-200 m benefit early: 48-52 % at save_9, vs 37-39 %.
- **Yaw (turret)**: no clear lead. 514 slightly ahead at save_9 (1.26-1.47 vs 1.51-1.88), then at 510's level afterwards.
- **At D=1000 (saves 20-23)**, 514 and 510 meet on yaw as on pitch, and 513 stays behind in pitch.

## Conclusion

- a·|a| and Φ change neither the final level (turret under 5°, precision at 600 k ≈ 510) nor the trigger.
- They seem however to **accelerate fine pitch precision**: about 1.0-1.2° from 243 k vs 1.3-1.8° for 510 and 513, with +10 points of hits at 50-200 m. Consistent with a·|a|, which gives fine control near zero on the canon, pitch being the axis where small corrections dominate.
- Limits: one seed per run, no seed-noise measure on this metric, and a·|a| and Φ confounded. Yaw gain not established.
