---
name: cd-release-pipeline
description: "Décisions validées du pipeline CD (2026-08-25) — cibles, gate version, ubuntu 22.04 + gcc-13, libtorch ARM via wheel, flag ARENAI_PORTABLE_RPATH"
metadata:
  node_type: memory
  type: project
  originSessionId: b0a908f7-c656-45fe-86a8-88eae5641e95
  modified: 2026-08-25T19:57:09.012Z
---

Pipeline CD ajouté à `.github/workflows/arenai-ci.yml` (2026-08-25), choix validés par Samuel :

- **3 cibles seulement** : linux x86_64, linux arm64, windows x86_64. **Pas de Windows ARM** (refusé).
- Zip = `arenai_desktop` **et** `arenai_agent_train` + `resources/`, nomenclature `arenai-[VERSION]_[OS]-[ARCH].zip` (version sans « v » ; le « v » n'apparaît que dans le tag de release).
- Gate : push master + CI verte + version.txt sans suffixe `snapshot` (insensible à la casse) + tag `v<version>` ≠ dernière release GitHub. Publie une release taguée `v<version>`.
- Linux : container **ubuntu:22.04** (glibc 2.35 = compat distros 2022+) + **gcc-13 via PPA ubuntu-toolchain-r** (std::format exige gcc ≥ 13). Build validé en docker local.
- Bundling `.so` dans `lib/` via `scripts/ci/package_linux.sh` (ldd + excludelist : glibc, X11/Wayland, libGL restent système ; libstdc++/libgcc sont bundlés). RPATH `$ORIGIN/lib:$ORIGIN` **uniquement** derrière le flag CMake `ARENAI_PORTABLE_RPATH` (OFF par défaut — Samuel ne veut pas de libs dupliquées en dev).
- LibTorch ARM : pas de zip officiel aarch64/win-arm64 → extraction du wheel pip `manylinux_2_28_aarch64` (`scripts/ci/fetch_libtorch.sh`, le dossier `torch/` du wheel = layout libtorch).
- Incohérence connue : LibTorch 2.12.0 sur Linux (CI env) vs 2.12.1 sur Windows (`install_dependencies.ps1`) — préexistante, signalée.

Voir aussi [[dev-environment]].

**Cible macOS arm64 ajoutée le 2026-11-21** (choix validés par Samuel, jamais exécutée sur un vrai Mac à cette date) :
- Job `package-macos` (runner `macos-14`) **seulement à la release**, **non bloquant** (`continue-on-error`, publish-release ne dépend que de Linux + Windows).
- Livrable = bundle `ArenAI.app` zippé (`scripts/ci/package_macos.sh`, dylibbundler, MoltenVK + ICD dans le bundle, signature ad hoc) ; pas de compte Apple Developer, donc `xattr -dr com.apple.quarantine` côté testeur.
- Samuel n'a pas de Mac : tests par des proches. Test local = compilation clang + libc++ en docker (script jetable), pas de macOS en docker.
- Samuel veut **éviter au maximum les branches par OS** (`#ifdef` linux/windows/macos) : préférer la détection de capacité ; exceptions acceptées : `__APPLE__` dans `arenai_desktop/src/main.cpp`, `if (APPLE)` pour le RPATH.
- `std::from_chars` flottant exige macOS 26 → remplacé par `std::strtof` ; minimum macOS = 13.3 côté code, 14 de fait (libs Homebrew du runner).
