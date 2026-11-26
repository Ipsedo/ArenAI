---
name: graphics-menu-screen
description: "Graphics submenu (shadows/MSAA/GPU) implemented 2026-08-29; GPU pickers as device-lists pending Samuel's visual OK; heightmap/cubemap selectors explicitly refused"
metadata:
  node_type: memory
  type: project
  originSessionId: 4aeed4f5-db26-4c72-b7f6-6517c1156d9c
  modified: 2026-08-29T19:59:46.514Z
---

Sous-écran Graphics du menu desktop, implémenté et mergé dans le working tree le
2026-08-29 (plan validé par Samuel) :

- Presets ombres Off/2048/4096/8192/Ultra 16384, MSAA 1/2/4/8, choix GPU fenêtre
  (effet au redémarrage) + GPU vision IA (effet au Play), persistés dans
  preferences.json (`shadow_quality`, `msaa`, `window_gpu`, `vision_gpu`).
- API publique ajoutée dans arenai_view/backend.h : `PlayerRendererSettings`,
  `gpu_name` sur les deux factories, `list_vulkan_gpus()`. Les env vars
  `ARENAI_VK_DEVICE*` gardent la priorité absolue (et throw si introuvables) ;
  le choix menu retombe en auto avec warning si le GPU a disparu.
- **Refusé par Samuel** : sélecteurs de heightmap/cubemap (retirés du plan à
  l'approbation) — ne pas les re-proposer sans demande. Aucun changement arenai_core.
- Étape restante : OK visuel de Samuel sur le mockup (GPU en `.device-list`
  vocabulaire sélecteur de manette, pas de dropdown RmlUi — le RCSS `select`
  n'existe pas dans menu.rcss), puis peaufinage éventuel du style, cf.
  [[menu-styling-validation]].
- Goldens vérifiés inchangés via goldens_docker.sh (défauts de rendu préservés :
  Ultra 16384 + MSAA 4 + GPU auto).


**2026-11-17 — mockup v2** (https://claude.ai/artifact/NSbo6rjNDCXzAFxS9eMrTN), awaiting Samuel's OK: adds VSync, post-process effects (3 checkboxes), resolution scale, fullscreen resolution, brightness + calibration card. Layout in 3 columns (Display: applies now / Quality: next Play / GPUs narrow 320dp). Toggle chips must become `flex: 1`, because the % widths only fit 544dp columns. Nothing implemented yet; adding these needs a `PlayerRendererSettings` contract change (validation required).

**2026-11-17, implemented** (not committed when written): mockup v2 validated, then ombres/MSAA changed to stepped sliders (mockup v3, validated). PlayerRendererSettings + render_scale/ambient_occlusion/bloom/sun_rays/brightness; AbstractWindowedGraphicBackend::set_vsync (MAILBOX → IMMEDIATE fallback); AbstractWindow::fullscreen_modes/set_fullscreen_mode. The Graphics page is only reachable from the main menu, so brightness applies on Play.
**RmlUi trap**: a `data-event-change` handler runs BEFORE `data-value` writes the bound variable, so pass `ev.value` as the event argument.
