---
name: core-tests-heightmap-cost
description: "arenai_core EnvironmentTest slowness (~104s) = Jolt MeshShape build of the heightmap at every reset, not Vulkan nor physics steps"
metadata:
  node_type: memory
  type: project
  originSessionId: 8ea1411b-813a-42b2-a6f0-be50107663ab
  modified: 2026-08-06T21:38:06.157Z
---

Profiling (perf, 2026-08-06): each `EnvironmentTest` costs 6–18 s and ~85–90 % of it is
`JPH::MeshShapeSettings`/`MeshShape` construction of the heightmap collider inside
`HeightMapItem::HeightMapItem` (`Indexify` 31 %, `sFindActiveEdges` 28 %, `TriangleSplitterBinning`
14 %, `Sanitize` 6 %), rebuilt by `BaseTanksEnvironment::reset_physics()` at every test. The 240
settle steps and the Vulkan backend are negligible (sharing the backend across tests via a static
in `EngineTestFixture` gained ~0). Same rebuild cost is paid at every training-episode reset.
**Fixed 2026-08-06** (Samuel's pick): the built `JPH::ShapeRefC` is now cached in a static
`utils::Cache<JPH::ShapeRefC>` in `height_map.cpp`, keyed by file path + scale — Samuel requires
his `arenai_utils` `Cache` class for this kind of caching, not an ad-hoc map. Core test binary:
104 s → ~8 s. Safe because the process-wide Jolt runtime in `jolt_engine.cpp` is never torn down.
`JPH::HeightFieldShape` remains a possible deeper cleanup (not validated). See [[dev-environment]]
for the build-dir trap found the same day.
