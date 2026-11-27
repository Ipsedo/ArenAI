# 556 save_24 — what the agent fires at when no living enemy is on the gun axis (2026-11-27)

Probe side 1000, 1 episode, 32 tanks, **deterministic**, binary at `5367a45e`, vision frames every 3 decisions
(`--frames`), joined to the decision CSV. Population: armed decisions (step ≥ 10) with the best-aligned living
enemy ≥ 20° off the gun axis. Control: 554 save_17 with its own `probe_bin`, same settings. Contact sheets (24
frames drawn at random, seed 0): `outputs/train_556_ppo_liquid/probes/save_24_offtarget_firing.png`,
`save_24_offtarget_not_firing.png`, `ref_554_save_17_offtarget_firing.png`. Frame dumps deleted (1 GB each).

| off-target armed decisions | 556 s24 | 554 s17 |
|---|---|---|
| all decisions of the episode: count, share firing | 10 731, 0.022 | 13 699, 0.011 |
| firing frames exported | 77 | 54 |
| firing: a living enemy somewhere in the camera field | 0.53 | 0.54 |
| firing: p_zoom, median \|μ_canon_x\| | 0.93, 0.06 | 0.86, 0.37 |
| not firing: p_zoom, median \|μ_canon_x\| | 0.75, 0.83 | — |

## What the frames show

- **Firing, 556 (24 frames)**: every frame has one salient object near the image centre, zoomed. 14 show a
  tank (several overturned or tilted), 10 show a map item (cylinder, cube, hemisphere). 5 of the 14 tank frames
  have no living enemy anywhere in the camera field.
- **Not firing, 556 (24 frames)**: bare terrain or sky in 19; an item in view in 5, off-centre or small.
- **Firing, 554 (24 frames)**: same content, 14 tanks and 10 items.

## Reading

The off-target trigger is not random: the agent stops the turret, zooms and fires at an object centred in the
image. Since the geometry excludes dead tanks and reports no living enemy within 20° of the axis, a tank
centred in the zoomed image is a dead one — inference from the geometry, the positions of dead tanks are not
exported. The other targets are map items. 554 does the same half as often (1.1 % vs 2.2 % of off-target
decisions) and with a softer posture.

Consequence for the game (not measured): game matches have no respawn, so wrecks stay for the whole match.

## Not established

That the centred tanks are wrecks (needs the bearing of dead tanks in the probe export); the share of the
in-game early dump carried by wrecks vs items; why 556 does it twice as often as 554 (one seed each).
