# Render environment for the golden-image tests. Source it, don't run it.
#
# Shared by the CI (.github/workflows/arenai-ci.yml) and scripts/goldens_docker.sh:
# the goldens are a byte-for-byte record of what this configuration renders, so
# the two must set exactly the same variables.

# Headless software rendering: the agent's vision is an offscreen EGL pbuffer
# and CI runners have no GPU.
export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe

# llvmpipe picks its vector width from the host CPU (AVX2 vs AVX-512), and that
# changes the rasterized pixels. GitHub runners have heterogeneous CPUs, so pin
# it to keep the goldens reproducible across machines.
export LP_NATIVE_VECTOR_WIDTH=256

# Tells the golden-image tests they are in the environment the references were
# recorded in, and may compare pixels. Everywhere else they skip that
# comparison instead of failing on the local Mesa/Bullet versions.
export ARENAI_PINNED_RENDER_ENV=1
