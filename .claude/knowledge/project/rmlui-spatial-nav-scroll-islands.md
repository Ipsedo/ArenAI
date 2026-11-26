---
name: rmlui-spatial-nav-scroll-islands
description: RmlUi 6.2 spatial nav (nav auto) never crosses scroll-container boundaries — bridged in C++ for the menu file explorer
metadata:
  node_type: memory
  type: project
  originSessionId: 36fe972c-ed30-45aa-ab93-505b124490d6
---

RmlUi 6.2's spatial navigation treats scroll containers (`overflow: auto/scroll`) as
isolated islands: `SearchNavigationTarget` (Source/Core/ElementDocument.cpp) skips their
subtree from outside, and `FindNextNavigationElement` restricts the search to the nearest
scroll-container ancestor from inside. `nav: auto` + `tab-index: auto` on the children is
NOT enough — D-pad focus can neither enter nor leave such a container.

Fix in place (2026-07-18): `ExplorerNavListener` in
`arenai_desktop/src/gui/rml_menu.cpp` bridges the `.file-list` borders on keydown
(anchor ids `#file-list`, `#display-row`, `#use-folder` in `parameters.rml`), and the
right stick scrolls the focused scroll container (`scroll_at_focus`). Any new scrollable
focusable region in the menu will need the same kind of bridge. See also
[[rmlui-vulkan-renderer-limits]].
