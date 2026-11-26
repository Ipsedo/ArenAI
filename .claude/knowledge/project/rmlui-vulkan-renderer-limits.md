---
name: rmlui-vulkan-renderer-limits
description: "ArenAI's Vulkan RmlUi render interface only supports basic geometry — box-shadow (and any layer/filter effect) renders as white quads + artifacts"
metadata:
  node_type: memory
  type: project
  originSessionId: 237bf2d8-662f-4b48-82d2-28b3f09dedfc
---

`arenai_view/src/vulkan/rml_render_interface.cpp` implements only the basic
Rml::RenderInterface surface (geometry, textures, scissor). It has none of the compositing
hooks (`PushLayer`/`PopLayer`, `SaveLayerAsTexture`, clip masks, callback textures) that
RmlUi 6 uses to bake `box-shadow`, filters and `backdrop-filter` into textures.

**Consequence:** any RCSS `box-shadow` (or filter) makes the element render as a plain
white quad, plus a stray white artifact at the screen origin (2026-07-18, focus-ring
incident). Menu styling must stick to plain geometry: `background-color`, `border`,
`border-radius`, colors. An HTML browser mockup will NOT catch this — browsers draw
shadows fine ([[menu-styling-validation]]).
