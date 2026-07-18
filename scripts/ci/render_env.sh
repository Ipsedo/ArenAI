# Render environment for the golden-image tests. Source it, don't run it.
#
# Shared by the CI (.github/workflows/arenai-ci.yml) and scripts/goldens_docker.sh:
# the goldens are a byte-for-byte record of what this configuration renders, so
# the two must set exactly the same variables.

# Headless software Vulkan: the agent's vision renders offscreen and CI
# runners have no GPU. lavapipe is Mesa's software Vulkan implementation
# (mesa-vulkan-drivers); both loader variable names are set, the older
# VK_ICD_FILENAMES for pre-1.3.207 loaders. The manifest name depends on the
# mesa package version (lvp_icd.x86_64.json before, lvp_icd.json now), hence
# the glob.
VK_ICD_FILENAMES="$(ls /usr/share/vulkan/icd.d/lvp_icd*.json | head -1)"
export VK_ICD_FILENAMES
export VK_DRIVER_FILES="$VK_ICD_FILENAMES"

# lavapipe shares the llvmpipe JIT, which picks its vector width from the host
# CPU (AVX2 vs AVX-512), and that changes the rasterized pixels. GitHub
# runners have heterogeneous CPUs, so pin it to keep the goldens reproducible
# across machines.
export LP_NATIVE_VECTOR_WIDTH=256

# Tells the golden-image tests they are in the environment the references were
# recorded in, and may compare pixels. Everywhere else they skip that
# comparison instead of failing on the local Mesa/Bullet versions.
export ARENAI_PINNED_RENDER_ENV=1
