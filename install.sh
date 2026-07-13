#!/usr/bin/env bash
#
# install.sh — build BASI-CLI from source on any Linux distro.
#
# Native binaries don't move across distros (glibc/ABI floors, absolute rpaths),
# so BASI installs by *compiling on your machine*. This script:
#   1. checks build dependencies (and can install them for you),
#   2. clones a pinned upstream llama.cpp and builds it with the Vulkan backend
#      (Vulkan runs on any NVIDIA/AMD/Intel GPU) into a persistent location,
#   3. compiles basi-cli against it,
#   4. drops a `basi` launcher on your PATH.
#
# Usage:
#   ./install.sh                          # clone+build llama.cpp, then BASI
#   ./install.sh --install-deps           # also install build deps via your pkg manager
#   ./install.sh --llama-dir ~/llama.cpp --llama-build ~/llama.cpp/build_vulkan
#                                         # reuse an existing llama.cpp build (skip clone)
#   ./install.sh --prefix ~/.local        # install prefix for the `basi` launcher
#   ./install.sh --jobs 8                 # parallel build jobs
#
set -euo pipefail

# ── pinned upstream llama.cpp ─────────────────────────────────────────────
# The commit BASI's llama-common / chat-template usage was written against
# (merge-base of the author's working branch with upstream master). Its `common`
# lib is internal and NOT ABI-stable, so pin it rather than tracking master.
LLAMA_REF="7dad2f1a17d65b5e2034c277125bc9f97573a779"
LLAMA_URL="https://github.com/ggml-org/llama.cpp.git"

# ── config (env-overridable; flags below take precedence) ─────────────────
PREFIX="${PREFIX:-$HOME/.local}"
SHARE="${SHARE:-$HOME/.local/share/basi}"
LLAMA_DIR="${LLAMA_DIR:-}"
LLAMA_BUILD="${LLAMA_BUILD:-}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
INSTALL_DEPS=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── pretty logging ────────────────────────────────────────────────────────
if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; D=$'\033[90m'; Z=$'\033[0m'
else B=; G=; Y=; R=; D=; Z=; fi
log()  { printf '%s==>%s %s\n' "$G$B" "$Z$B" "$*$Z"; }
info() { printf '%s  - %s%s\n' "$D" "$*" "$Z"; }
warn() { printf '%swarning:%s %s\n' "$Y$B" "$Z" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$R$B" "$Z" "$*" >&2; exit 1; }

# ── flags ─────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix)       PREFIX="$2"; shift 2;;
    --share)        SHARE="$2"; shift 2;;
    --llama-dir)    LLAMA_DIR="$2"; shift 2;;
    --llama-build)  LLAMA_BUILD="$2"; shift 2;;
    --llama-ref)    LLAMA_REF="$2"; shift 2;;
    --jobs|-j)      JOBS="$2"; shift 2;;
    --install-deps) INSTALL_DEPS=1; shift;;
    -h|--help)      sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) die "unknown option: $1 (see --help)";;
  esac
done

# ── dependency check ──────────────────────────────────────────────────────
detect_pkgmgr() {
  local m
  for m in pacman apt-get dnf zypper; do
    command -v "$m" >/dev/null 2>&1 && { echo "$m"; return; }
  done
  echo ""
}
install_cmd() {
  case "$1" in
    pacman)  echo "sudo pacman -S --needed base-devel cmake git shaderc vulkan-headers vulkan-icd-loader curl";;
    apt-get) echo "sudo apt-get install -y build-essential cmake git glslc libvulkan-dev curl";;
    dnf)     echo "sudo dnf install -y gcc-c++ make cmake git glslc vulkan-headers vulkan-loader-devel curl";;
    zypper)  echo "sudo zypper install -y gcc-c++ make cmake git shaderc vulkan-headers vulkan-devel libvulkan1 curl";;
    *)       echo "";;
  esac
}
have_vulkan_headers() {
  pkg-config --exists vulkan 2>/dev/null && return 0
  [ -f /usr/include/vulkan/vulkan.h ] && return 0
  return 1
}
have_vulkan_loader() {
  ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so\.1' && return 0
  ls /usr/lib*/libvulkan.so.1 /usr/lib/*/libvulkan.so.1 >/dev/null 2>&1 && return 0
  return 1
}

check_deps() {
  log "Checking build dependencies"
  local missing=() t
  for t in gcc g++ make cmake git curl glslc; do
    command -v "$t" >/dev/null 2>&1 || missing+=("$t")
  done
  have_vulkan_headers || missing+=("vulkan-headers")
  have_vulkan_loader  || missing+=("vulkan-loader(libvulkan.so.1)")

  if [ ${#missing[@]} -eq 0 ]; then
    info "all present"
    return 0
  fi

  warn "missing: ${missing[*]}"
  local pm cmd; pm="$(detect_pkgmgr)"; cmd="$(install_cmd "$pm")"
  if [ -z "$cmd" ]; then
    die "install these with your package manager, then re-run. (needed: a C/C++ toolchain, cmake, git, curl, glslc, and the Vulkan headers + loader)"
  fi
  if [ "$INSTALL_DEPS" -eq 1 ]; then
    log "Installing dependencies: $cmd"
    eval "$cmd"
  else
    die "run:  $cmd
   ...or re-run this script with --install-deps to do it automatically."
  fi
}

# ── build llama.cpp (pinned, Vulkan, shared libs) ─────────────────────────
build_llama() {
  if [ -n "$LLAMA_DIR" ]; then
    [ -d "$LLAMA_DIR" ] || die "--llama-dir '$LLAMA_DIR' does not exist"
    info "reusing llama.cpp checkout: $LLAMA_DIR"
  else
    LLAMA_DIR="$SHARE/llama.cpp"
    if [ ! -e "$LLAMA_DIR/.git" ]; then
      log "Cloning llama.cpp -> $LLAMA_DIR"
      mkdir -p "$SHARE"
      git clone --filter=blob:none "$LLAMA_URL" "$LLAMA_DIR" \
        || git clone "$LLAMA_URL" "$LLAMA_DIR"
    fi
    log "Pinning llama.cpp @ ${LLAMA_REF:0:12}"
    git -C "$LLAMA_DIR" fetch --depth 1 origin "$LLAMA_REF" 2>/dev/null \
      || git -C "$LLAMA_DIR" fetch origin
    git -C "$LLAMA_DIR" checkout -q "$LLAMA_REF"
  fi

  LLAMA_BUILD="${LLAMA_BUILD:-$LLAMA_DIR/build_vulkan}"
  if [ -f "$LLAMA_BUILD/bin/libllama-common.so" ]; then
    info "reusing existing build: $LLAMA_BUILD"
    return 0
  fi
  log "Building llama.cpp (Vulkan) -> $LLAMA_BUILD  [$JOBS jobs]"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_CURL=OFF
  cmake --build "$LLAMA_BUILD" -j"$JOBS"
  [ -f "$LLAMA_BUILD/bin/libllama-common.so" ] \
    || die "llama.cpp build finished but libllama-common.so is missing in $LLAMA_BUILD/bin"
}

# ── build BASI against the llama.cpp we just prepared ─────────────────────
build_basi() {
  log "Building basi-cli  [$JOBS jobs]"
  make -C "$REPO_ROOT" clean >/dev/null 2>&1 || true
  make -C "$REPO_ROOT" -j"$JOBS" LLAMA_DIR="$LLAMA_DIR" LLAMA_BUILD="$LLAMA_BUILD"
  [ -x "$REPO_ROOT/basi-cli" ] || die "build reported success but basi-cli is missing"
}

# ── install binary + launcher ─────────────────────────────────────────────
install_files() {
  log "Installing"
  mkdir -p "$SHARE" "$PREFIX/bin"
  install -m755 "$REPO_ROOT/basi-cli" "$SHARE/basi-cli"
  info "binary -> $SHARE/basi-cli"

  # The binary's rpath points at $LLAMA_BUILD/bin (absolute), so the launcher is
  # a thin exec that just preserves the current working directory as the project
  # root — the same convention `claude` uses. No model is hardcoded: BASI resolves
  # -m > /model saved default > \$BASI_MODEL > interactive picker.
  cat > "$PREFIX/bin/basi" <<EOF
#!/usr/bin/env bash
# basi — run BASI-CLI from any directory; operates on the current working dir.
exec "$SHARE/basi-cli" "\$@"
EOF
  chmod +x "$PREFIX/bin/basi"
  info "launcher -> $PREFIX/bin/basi"
}

# ── go ────────────────────────────────────────────────────────────────────
check_deps
build_llama
build_basi
install_files

echo
log "BASI installed."
case ":$PATH:" in
  *":$PREFIX/bin:"*) : ;;
  *) warn "$PREFIX/bin is not on your PATH. Add it:"
     printf '     %sexport PATH="%s/bin:$PATH"%s   # add to ~/.bashrc\n' "$B" "$PREFIX" "$Z";;
esac
cat <<EOF

  Next:
    ${B}basi${Z}                 run it in any project directory
    ${B}/cookbook${Z}            (inside basi) download a GGUF model to ~/models
    ${B}/model <name>${Z}        switch models

  Note: needs a Vulkan-capable GPU driver at runtime — Mesa (vulkan-radeon /
  vulkan-intel) for AMD/Intel, or the proprietary NVIDIA driver. CPU-only works
  too but is slow.
EOF
