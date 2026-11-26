---
name: menu-styling-validation
description: "Menu charter history: current charter is 'Duel néon' (menu.rcss header), Blindage is obsolete; how to preview a mockup"
metadata:
  node_type: memory
  type: project
  originSessionId: 19ade219-0fec-4630-a2ac-609174dabf7b
---

Samuel angrily rejected a focus-ring color (sun/amber #ffd27a) I applied to the RmlUi menus
without showing it first — even though the hex was technically in the charter comment, the
combination was not part of the site's visual language.

**Why:** Samuel cares deeply about visual coherence; a token being listed in menu.rcss's
palette comment does not mean any use of it fits.

Current charter (validated 2026-07-18, replaced the ipsedo.com theme): "Blindage bleu
profond" — structure Blindage (coins 2–3dp, liseré gauche 3dp accent, titres/boutons en
capitales espacées) sur la palette coolors #00A6FB / #0582CA / #006494 / #003554 / #051923,
avec teintes claires dérivées de l'accent pour le texte (encre #EAF7FF, labels #8AC4E0,
sélection #7FD3FF, disabled #6FA3BF) — Samuel sait qu'elles sont dérivées et ajustables.
The mockup artifact lives at https://claude.ai/code/artifact/15da417f-154b-435e-b917-aef73d1d9cc7
(source in the session scratchpad: menu-styles.template.html + build_mockups.py).
Note: the cursor-ring decorator (rml_menu.cpp) now follows the element's border-radius
instead of hardcoding a circle — the RCSS hover rule must set border-radius (6dp) on
sliderbar for the concentric square-ish ring.


**Update 2026-11-17:** the Blindage charter is obsolete. The current charter is "Duel néon" (player orange #FF7A2F vs AI cyan #2FE6FF, slant parallelograms, Chakra Petch + JetBrains Mono); its tokens are in the header of `resources/menu/menu.rcss` (not arenai_desktop/resources). Reference mockups: Design canvas https://claude.ai/artifact/2uxbQJZxXztSQdmFhEAH6J (C_* boards). Local file previews can't be screenshotted and artifact URLs require sign-in in the browser pane: serve the scratchpad over `python -m http.server` on 127.0.0.1 to check a mockup.
