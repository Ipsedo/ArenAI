# Changelog

Notable changes per released version; unreleased work is not listed. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow the git tags.

## [1.2.6] - 2026-09-27

### Changed

- Vanilla PPO back to a truncated normal distribution (Beta needs a very high concentration).

### Fixed

- Continuous action inference of vanilla PPO.

## [1.2.5] - 2026-09-22

### Added

- `ppo_liquid` agent (liquid time-constant recurrent cell).
- Zoom action and Bernoulli head for discrete actions.
- Spawn-distance curriculum driven by the hit ratio (up to 2000 m).
- More objects on the map.
- Hyperparameters saved and loaded as JSON alongside the checkpoint.

### Changed

- Starvation ends the episode as a truncation, without penalty; win reward only with at least one
  kill.

### Removed

- SAC agent.

### Fixed

- Turret oscillation, object spawn raycast, gamepad navigation in the file explorer.

## [1.2.4] - 2026-09-02

### Added

- Hit marker for the player, controls and graphics configuration menus.
- Training configuration logged with each run.

### Changed

- Entropy regulated by a PID Lagrangian towards a constant target (annealing dropped).
- Hit reward $1/(1+r^2)$; enemy fire cooldown removed; longer horizon ($\gamma$, $\lambda$).

### Fixed

- Truncated normal mean at inference.

## [1.2.3] - 2026-08-25

### Changed

- Release archive naming.

## [1.2.2] - 2026-08-25

### Changed

- Hit reward tuning.

## [1.2.1] - 2026-08-25

### Added

- Reticle and score on the pause menu.

## [1.2.0] - 2026-08-21

### Added

- Shell reserve with regeneration; penalty on received hits.
- One entropy temperature per continuous action; PID Lagrangian entropy control for PPO.
- Heightmap loading cache.

### Changed

- Hit reward from the miss distance and angle.
- PPO: the KL early stop no longer stops the critic.

### Fixed

- Damage on shell contact, SAC target (expectation of the min), normalisation layers in eval mode.

## [1.1.2] - 2026-07-31

### Added

- PPO agent, end-to-end tests, game-over popup.

### Changed

- Physics engine Bullet → Jolt.
- Training module renamed `arenai_agent`, LibTorch encapsulated.
- Hit reward from the shell's closest approach to a tank.

## [1.1.1] - 2026-07-19

### Changed

- Menu style, gamepad navigation in menus, file explorer to pick a checkpoint.

### Removed

- ExecuTorch conversion.

## [1.1.0] - 2026-07-18

### Added

- RmlUi menu, Xbox gamepad support, dynamic shadows, post-processing, spring camera.
- GitHub Actions CI with golden image tests.

### Changed

- Renderer GLES → OpenGL → Vulkan; shaders embedded; backend-independent view headers.

### Removed

- Android application.

## [1.0.0] - 2026-07-02

### Added

- Battle-royale tank game in a physics world; SAC agent trained on its camera frame; desktop game
  and Android app.

[1.2.6]: https://github.com/Ipsedo/ArenAI/compare/v1.2.5...v1.2.6
[1.2.5]: https://github.com/Ipsedo/ArenAI/compare/v1.2.4...v1.2.5
[1.2.4]: https://github.com/Ipsedo/ArenAI/compare/v1.2.3...v1.2.4
[1.2.3]: https://github.com/Ipsedo/ArenAI/compare/v1.2.2...v1.2.3
[1.2.2]: https://github.com/Ipsedo/ArenAI/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/Ipsedo/ArenAI/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/Ipsedo/ArenAI/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/Ipsedo/ArenAI/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/Ipsedo/ArenAI/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/Ipsedo/ArenAI/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Ipsedo/ArenAI/releases/tag/v1.0.0
