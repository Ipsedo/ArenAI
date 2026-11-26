---
name: hud-hitmarker-battlefield
description: "Hit marker HUD validé style Battlefield (2026-08-23) — ticks fins qui s'écartent, kill flare rouge, sans texte de score ; SetTransform ajouté au backend RmlUi Vulkan"
metadata:
  node_type: memory
  type: project
  originSessionId: 1d2eecad-09f2-4dad-bb4c-95ea71cafe94
  modified: 2026-08-27T21:24:14.825Z
---

Le 2026-08-23, Samuel a validé (via maquette artifact comparant CoD/BF/ArenAI) le style
**Battlefield** pour le hit marker du HUD, explicitement **sans le texte de score au kill**.

Le réticule complet est passé au style BF (retour du 2026-08-23 : « pas comme la maquette,
et le point central ? ») : croix fine 1.5dp, boîte 28dp, gap 5dp + point central Ø2.5dp
(#crosshair-dot, quad arrondi) — le X du marqueur a le même rayon que la croix, comme sur
la maquette. Ne pas revenir à l'ancienne croix 48dp/2dp sans validation.

Implémentation : `hud.rcss` (#hit-marker : 30dp, 2dp, gap 11dp), `rml_menu.cpp::notify_hit`
(opacité + échelle ease-out quad : hit 450 ms 1→1.3 ; kill 600 ms 1.1→1.55, rouge #FF3B3B),
et `SetTransform` implémenté dans `RmlVulkanRenderInterface` (le backend l'ignorait avant —
toute animation `transform` RmlUi était invisible sans ça ; push = projection * transform).

**Non retenu / ouvert** : pas de son de hit (BF/CoD en ont un ; ArenAI n'a pas de canal audio),
pas de palier armure. Cf. [[menu-styling-validation]] pour la règle maquette-avant-style.

**Ajout 2026-08-27 — arc de dégâts entrants (option B validée sur maquette)** : quand le
joueur est touché, arc annulaire rouge #FF3B3B autour du centre écran orienté vers le tireur,
fondu 0,8 s, pool de 4 slots. Direction = position d'impact − centre châssis (choix de Samuel :
impact, PAS la position de tir), convertie en angle écran via la base caméra. Canal modèle :
struct `ImpactInfo{vec3 position; float damages}` dans item.h, `LifeItem::hits_received` est
devenu un vecteur d'ImpactInfo (compte = .size(), consommé par le reward ennemi → borné),
`PlayerTank::consume_received_impacts()`. Décorateur `damage-arc` (secteur annulaire, alpha
par vertex, pointes fondues) + rotation via transform. Vignette (options A/C) non retenue à ce
stade ; réglages 100dp/16dp/60° à ajuster à l'œil.
