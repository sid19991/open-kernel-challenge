#!/bin/sh
# Install okc, the Open Kernel Challenge CLI.
#
#   curl -fsSL https://openkernelchallenge.dev/install.sh | sh
#
# OKC_VERSION picks a version (default: latest), OKC_INSTALL_DIR where it goes
# (default: ~/.okc/bin). The download is checked against the release's SHA256SUMS. When the
# directory isn't on PATH, one line adding it goes into the shell's startup file (marked, and
# added once); OKC_NO_MODIFY_PATH=1 skips that.
set -eu
BASE_URL=${OKC_BASE_URL:-https://openkernelchallenge.dev/okc}
DIR=${OKC_INSTALL_DIR:-$HOME/.okc/bin}

# Errors before okc is in place, and the plain report for an old okc; okc draws the rest.
ok() { printf '✓ %s\n' "$*"; }
warn() { printf '! %s\n' "$*"; }
fail() { printf '✗ %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "this installer needs $1"; }
tilde() { case $1 in "$HOME"/*) printf '~%s' "${1#"$HOME"}" ;; *) printf '%s' "$1" ;; esac; }
need curl
need uname

case "$(uname -s)" in
  Darwin) os=darwin; os_name=macOS ;;
  Linux) os=linux; os_name=Linux ;;
  *) fail "okc runs on macOS and Linux, not $(uname -s)" ;;
esac
case "$(uname -m)" in
  arm64 | aarch64) arch=arm64 ;;
  x86_64 | amd64) arch=amd64 ;;
  *) fail "no okc build for $(uname -m) CPUs" ;;
esac

version=${OKC_VERSION:-$(curl -fsSL "$BASE_URL/latest")} || fail "can't reach $BASE_URL; check your connection"
bin="okc-$os-$arch"
tmp=$(mktemp -d)
pid=
trap 'rm -rf "$tmp"' EXIT
# A download started in the background ignores Ctrl-C, so stop it ourselves.
trap '[ -n "$pid" ] && kill "$pid" 2>/dev/null; printf "\n" >&2; exit 130' INT TERM

# progress label file: one line, redrawn in place, of how much of file has arrived out of the
# Content-Length curl saved (the last one, after redirects), or just the amount before that.
progress() {
  got=$(wc -c 2>/dev/null < "$2") || got=0
  total=$({ tr -d '\r' < "$tmp/headers"; } 2>/dev/null | awk 'tolower($1) == "content-length:" { n = $2 } END { print n + 0 }')
  awk -v l="$1" -v g="${got:-0}" -v t="$total" 'BEGIN {
    mb = 1048576; w = 24
    if (t <= 0) { printf "\r\033[K  %s  %.1f MB", l, g / mb; exit }
    n = int(w * g / t); if (n > w) n = w
    bar = ""; for (i = 0; i < n; i++) bar = bar "━"
    rest = ""; for (i = n; i < w; i++) rest = rest "━"
    printf "\r\033[K  %s  %s\033[2m%s\033[0m  %3d%%  %.1f / %.1f MB", l, bar, rest, 100 * g / t, g / mb, t / mb
  }' >&2
}
# fetch url file label: curl, with a progress line when there's a terminal to draw it on.
fetch() {
  [ -t 2 ] || { curl -fsSL "$1" -o "$2"; return; }
  curl -fsSL -D "$tmp/headers" "$1" -o "$2" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do progress "$3" "$2"; sleep 0.1; done
  status=0
  wait "$pid" || status=$?
  pid=
  printf '\r\033[K' >&2
  return "$status"
}

curl -fsSL "$BASE_URL/$version/SHA256SUMS" -o "$tmp/SHA256SUMS" || fail "couldn't download okc $version's checksums"
# Gzipped is under half the size; releases before 0.6.5 have only the plain binary.
if command -v gunzip >/dev/null 2>&1 && fetch "$BASE_URL/$version/$bin.gz" "$tmp/$bin.gz" "downloading okc $version"; then
  size=$(wc -c < "$tmp/$bin.gz")
  gunzip -c "$tmp/$bin.gz" > "$tmp/$bin" || fail "couldn't unpack okc $version"
else
  fetch "$BASE_URL/$version/$bin" "$tmp/$bin" "downloading okc $version" || fail "couldn't download okc $version for $os_name $arch"
  size=$(wc -c < "$tmp/$bin")
fi
size=$(printf '%s' "$size" | awk '{ printf "%.1f MB", $1 / 1048576 }')

want=$(grep " $bin\$" "$tmp/SHA256SUMS" | cut -d' ' -f1)
[ -n "$want" ] || fail "no checksum for $bin in okc $version's SHA256SUMS"
if command -v sha256sum >/dev/null 2>&1; then got=$(sha256sum "$tmp/$bin" | cut -d' ' -f1)
else got=$(shasum -a 256 "$tmp/$bin" | cut -d' ' -f1); fi
[ "$got" = "$want" ] || fail "checksum mismatch: expected $want, got $got; nothing was installed"
sha=$(printf '%s' "$got" | cut -c1-12)

mkdir -p "$DIR"
mv "$tmp/$bin" "$DIR/okc"
chmod +x "$DIR/okc"

# On PATH already, or added to the shell's startup file once.
added= missing= line=
case ":$PATH:" in
  *":$DIR:"*) ;;
  *)
    # $HOME kept as written, so the line is short and survives a moved home directory.
    case $DIR in "$HOME"/*) shown="\$HOME${DIR#"$HOME"}" ;; *) shown=$DIR ;; esac
    line="export PATH=\"$shown:\$PATH\""
    case "${SHELL:-}" in
      */zsh) rc="${ZDOTDIR:-$HOME}/.zshrc" ;;
      */bash) if [ "$os" = darwin ]; then rc="$HOME/.bash_profile"; else rc="$HOME/.bashrc"; fi ;;
      */fish) rc="$HOME/.config/fish/conf.d/okc.fish"; line="fish_add_path $shown" ;;
      *) rc= ;;
    esac
    missing=$(tilde "$DIR")
    if [ -z "${OKC_NO_MODIFY_PATH:-}" ] && [ -n "$rc" ]; then
      if ! grep -qs "# okc" "$rc"; then
        mkdir -p "$(dirname "$rc")"
        printf '\n%s  # okc\n' "$line" >> "$rc"
      fi
      added=$(tilde "$rc")
    fi
    ;;
esac

# The report, drawn by okc itself so it matches the CLI; okc before 0.6.3 has no _installed,
# so this prints it plainly instead.
set -- --platform "$os_name $arch" --size "$size" --sha "$sha" --path "$(tilde "$DIR/okc")"
[ -n "$missing" ] && set -- "$@" --missing "$missing" --line "$line"
[ -n "$added" ] && set -- "$@" --added "$added"
# okc asks the terminal for its background (light or dark green, as the CLI does) and reads
# the answer from its input: under `curl | sh` that input is this script, so it gets the
# terminal itself, or nothing when there isn't one.
if [ -t 1 ] && { : </dev/tty; } 2>/dev/null; then tty=/dev/tty; else tty=/dev/null; fi
if ! "$DIR/okc" _installed "$@" <"$tty" 2>/dev/null; then
  printf '\nokc · open kernel challenge · muse spark open\n  meta × paradigm × andromeda\n\n'
  ok "$os_name $arch"
  ok "downloaded okc $version ($size)"
  ok "checksum verified (sha256 $sha)"
  ok "installed to $(tilde "$DIR/okc")"
  if [ -n "$added" ]; then ok "added $missing to your PATH ($added)"; warn "open a new terminal to use okc, or run"; printf '    %s\n' "$line"
  elif [ -n "$missing" ]; then warn "$missing isn't on your PATH; add it with"; printf '    %s\n' "$line"; fi
  printf '\nnext\n  okc login            sign in with GitHub\n  okc problems         the kernels to beat SGLang on\n  okc init <problem>   the starter and SGLang'"'"'s kernel\n\n'
fi
