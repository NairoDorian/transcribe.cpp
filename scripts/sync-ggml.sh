#!/usr/bin/env bash
#
# sync-ggml.sh — re-vendor ggml/ from upstream ggml-org/ggml at a given ref.
#
# ggml is vendored (not a submodule): ggml/ is the upstream tracked tree at the
# SHA recorded in ggml/UPSTREAM, plus the ordered downstream patches in
# patches/ggml/. The SHA and patch directory are the complete vendor recipe.
#
# What it does:
#   1. Fetches the upstream tracked tree at <ref> (a SHA, tag, or branch).
#   2. Materializes it via `git archive` (tracked files only — no .git, no
#      build cruft), minus the paths in EXCLUDES below.
#   3. Applies patches/ggml/*.patch in filename order.
#   4. Swaps the result into ggml/ and rewrites ggml/UPSTREAM.
#
# Upstream examples and tests are kept (they are not built — TRANSCRIBE_*/
# GGML_BUILD_* leave them off — but keeping them makes the vendor diff
# reviewable). Only the paths in EXCLUDES below are dropped, each for the
# reason recorded beside it.
#
# Re-vendoring the CURRENT pinned SHA must be a no-op. The script enforces that
# (`sync + patches == committed tree`): with no ref given, any path under ggml/
# that the recipe does not reproduce in the committed tree is a reproducibility
# failure — a hand-edit to the vendored tree or a patch that does not apply
# cleanly — and exits non-zero. Pass a different ref to upgrade, or --force to
# accept a repair.
#
# Usage:
#   scripts/sync-ggml.sh                 # re-vendor the CURRENT pinned SHA (repair / verify)
#   scripts/sync-ggml.sh master          # bump to upstream default-branch HEAD
#   scripts/sync-ggml.sh <sha|tag>       # pin to a specific commit or release tag
#   scripts/sync-ggml.sh master --dry-run   # show what would change, write nothing
#
# Flags:
#   --dry-run        Resolve the ref and report the file-level diff; do not touch ggml/.
#   --check          Like --dry-run, but exit non-zero if the recipe does NOT
#                    reproduce the committed ggml/. This is the CI gate for
#                    `sync + patches == tree`; it writes nothing either way.
#   --force          Proceed even if ggml/ has uncommitted local changes (default: abort,
#                    so an accidental hand-edit is never silently clobbered). Also
#                    downgrades a no-op-check failure to a warning.
#   --repo <url>     Override the upstream URL (default: the repo: line in ggml/UPSTREAM).
#
# Exit-code driven, non-interactive. After a real sync, build and let native-ci
# (path filter ggml/**) certify the C/C++ contracts the bindings depend on:
#   cmake --build build --target transcribe-cli

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGML_DIR="${REPO_ROOT}/ggml"
UPSTREAM_FILE="${GGML_DIR}/UPSTREAM"
PATCH_DIR="${REPO_ROOT}/patches/ggml"

# Upstream paths to drop from the snapshot (relative to the ggml tree root).
# Dropping a path here is a deliberate, recorded act — that is the whole point of
# the list, and of the check below: a path that vanishes from the vendored tree
# without a line here, or a patch, is the silent loss this scheme prevents.
#   .github          upstream CI — irrelevant here.
#   .pi              a symlink into the packager's local config dir.
#   .gitmodules      a 0-byte residue that would otherwise read as "ggml is a
#                    submodule" to whoever finds it first.
#   examples/mnist/web   upstream's own web/.gitignore is `*`, so git will not
#                    add the directory's contents (nor the .gitignore itself)
#                    without --force — a vendoring commit cannot carry it. Held
#                    out of the recipe rather than force-added, so the recipe
#                    keeps describing what is actually committed.
EXCLUDES=( ".github" ".pi" ".gitmodules" "examples/mnist/web" )

shopt -s nullglob
PATCHES=( "${PATCH_DIR}"/*.patch )
shopt -u nullglob

# ---- parse args -------------------------------------------------------------
REF=""
REPO=""
DRY_RUN=0
CHECK=0
FORCE=0

die() { echo "sync-ggml: $*" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --check)   DRY_RUN=1; CHECK=1; shift ;;
        --force)   FORCE=1; shift ;;
        --repo)    REPO="${2:-}"; [ -n "$REPO" ] || die "--repo needs a URL"; shift 2 ;;
        -h|--help) sed -n '2,46p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --*)       die "unknown flag: $1" ;;
        *)         [ -z "$REF" ] || die "more than one ref given ($REF, $1)"; REF="$1"; shift ;;
    esac
done

[ -f "$UPSTREAM_FILE" ] || die "missing $UPSTREAM_FILE — run from a checkout with vendored ggml"

# ---- read current pin -------------------------------------------------------
CUR_REPO="$(sed -n 's/^repo:[[:space:]]*//p' "$UPSTREAM_FILE" | tr -d '\r' | head -1)"
CUR_SHA="$(sed -n 's/^sha:[[:space:]]*//p'  "$UPSTREAM_FILE" | tr -d '\r' | head -1)"
[ -n "$CUR_REPO" ] || die "no 'repo:' line in $UPSTREAM_FILE"
[ -n "$CUR_SHA"  ] || die "no 'sha:' line in $UPSTREAM_FILE"

REPO="${REPO:-$CUR_REPO}"
REF="${REF:-$CUR_SHA}"   # default: re-vendor the current pin (idempotent repair)

command -v git >/dev/null || die "git not found"
command -v tar >/dev/null || die "tar not found"

# ---- guard against clobbering local edits -----------------------------------
if [ "$DRY_RUN" -eq 0 ] && [ "$FORCE" -eq 0 ]; then
    if [ -n "$(git -C "$REPO_ROOT" status --porcelain -- ggml 2>/dev/null)" ]; then
        die "ggml/ has uncommitted changes. Commit/stash them, or pass --force to overwrite."
    fi
fi

# ---- scratch dirs (self-cleaning) -------------------------------------------
CLONE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sync-ggml.XXXXXX")"
STAGE_DIR="${REPO_ROOT}/ggml.sync-stage.$$"   # sibling of ggml/ → same FS → atomic mv
REF_DIR=""            # set below: HEAD:ggml materialized, the comparison basis
cleanup() {
    rm -rf "$CLONE_DIR" "$STAGE_DIR"
    [ -z "$REF_DIR" ] || rm -rf "$REF_DIR"
}
trap cleanup EXIT

# ---- fetch just the requested commit ----------------------------------------
echo "sync-ggml: fetching $REF from $REPO ..."
git init  --quiet "$CLONE_DIR"
git -C "$CLONE_DIR" remote add origin "$REPO"
# `git archive` is a checkout-shaped operation: it applies the working-tree
# line-ending conversion, so it inherits the clone's core.autocrlf. Git for
# Windows sets that to true at *system* level — and the `git init` above
# inherits it — so without this the staged tree materializes CRLF on Windows
# and LF on Linux, while every committed blob is LF (`.gitattributes`
# normalizes on the way in). The recipe's output would then depend on the
# machine that ran it, which no reproducibility gate can be built on.
git -C "$CLONE_DIR" config core.autocrlf false
# `git fetch <ref>` resolves a SHA, tag, or branch in one shot; --depth 1 keeps
# it to a single commit's complete tree+blobs (GitHub allows arbitrary-SHA fetch).
git -C "$CLONE_DIR" fetch --quiet --depth 1 origin "$REF" \
    || die "could not fetch '$REF' (try a branch/tag, or check the SHA is reachable)"
# Peel to the commit: for an annotated tag, FETCH_HEAD is the tag *object* —
# ^{commit} resolves it to the commit SHA (a no-op for branch/commit refs), so
# UPSTREAM records a real commit, matching the existing convention.
RESOLVED="$(git -C "$CLONE_DIR" rev-parse "FETCH_HEAD^{commit}")"

# ---- materialize tracked tree, minus excludes -------------------------------
mkdir -p "$STAGE_DIR"
EXCLUDE_ARGS=()
for ex in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=( "--exclude=${ex}" "--exclude=./${ex}" )
done
git -C "$CLONE_DIR" archive --format=tar "$RESOLVED" | tar -x "${EXCLUDE_ARGS[@]}" -C "$STAGE_DIR"
for ex in "${EXCLUDES[@]}"; do
    rm -rf "${STAGE_DIR:?}/${ex}"
done

# ---- materialize the committed tree, the comparison basis -------------------
# What is checked is `sync + patches == committed tree`, so the tree on the
# other side of the comparison is HEAD:ggml, NOT the working copy. The working
# copy is a platform-dependent materialization of it: on Windows anything
# `.gitattributes` does not pin to eol=lf comes out CRLF (text=auto plus the
# Git-for-Windows default core.autocrlf=true), and a ggml/ last written by an
# older run of this script is CRLF throughout — 2240 of its 2241 files, since
# the swap below is a plain `mv` that bypasses git's filters. Comparing bytes
# against that would report the end-of-line convention as drift on one
# platform, hide real drift on another (a working copy newer than the commit
# makes the check green while HEAD is wrong — the one thing the check exists
# for), and make --check disagree with CI, which checks out clean and so
# compares against the commit either way.
#
# Conversion off on both sides, and the same EXCLUDES the stage drops, so the
# two trees describe the same path set and a byte difference means drift.
if git -C "$REPO_ROOT" rev-parse --verify --quiet HEAD:ggml >/dev/null; then
    REF_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sync-ggml-head.XXXXXX")"
    git -C "$REPO_ROOT" -c core.autocrlf=false archive --format=tar HEAD:ggml \
        | tar -x "${EXCLUDE_ARGS[@]}" -C "$REF_DIR"
    for ex in "${EXCLUDES[@]}"; do
        rm -rf "${REF_DIR:?}/${ex}"
    done
fi

# ---- apply downstream patches ----------------------------------------------
STAGE_NAME="$(basename "$STAGE_DIR")"
for patch in "${PATCHES[@]}"; do
    echo "sync-ggml: applying patches/ggml/$(basename "$patch")"
    git -C "$REPO_ROOT" apply --check --directory="$STAGE_NAME" "$patch" \
        || die "patch does not apply: patches/ggml/$(basename "$patch")"
    git -C "$REPO_ROOT" apply --directory="$STAGE_NAME" "$patch"
done

# ---- regenerate UPSTREAM ----------------------------------------------------
# Read the version out of the staged tree rather than the clone so the recorded
# value describes what actually landed (a patch could, in principle, move it).
GGML_VERSION=""
if [ -f "${STAGE_DIR}/CMakeLists.txt" ]; then
    _maj="$(sed -n 's/^set(GGML_VERSION_MAJOR[[:space:]]*\([0-9]*\)).*/\1/p' "${STAGE_DIR}/CMakeLists.txt" | tr -d '\r' | head -1)"
    _min="$(sed -n 's/^set(GGML_VERSION_MINOR[[:space:]]*\([0-9]*\)).*/\1/p' "${STAGE_DIR}/CMakeLists.txt" | tr -d '\r' | head -1)"
    _pat="$(sed -n 's/^set(GGML_VERSION_PATCH[[:space:]]*\([0-9]*\)).*/\1/p' "${STAGE_DIR}/CMakeLists.txt" | tr -d '\r' | head -1)"
    [ -n "$_maj" ] && GGML_VERSION="${_maj}.${_min:-0}.${_pat:-0}"
fi
GGML_DATE="$(git -C "$CLONE_DIR" show -s --format=%cs "$RESOLVED" 2>/dev/null || echo unknown)"

{
    cat <<EOF
repo: ${REPO}
sha:  ${RESOLVED}
version: ${GGML_VERSION:-unknown}
date: ${GGML_DATE}
patches:
EOF
    if [ "${#PATCHES[@]}" -eq 0 ]; then
        echo "  (none)"
    else
        for patch in "${PATCHES[@]}"; do
            echo "  patches/ggml/$(basename "$patch")"
        done
    fi
    cat <<'EOF'

The `version` and `date` lines are conveniences for humans reviewing a pin bump:
they are read from the vendored ggml/CMakeLists.txt and the upstream commit
date, so a bare SHA does not have to be decoded by hand.

This directory is generated from the upstream ggml tree at the SHA above, minus
the paths in the sync script's EXCLUDES, with the listed downstream patches
applied in order. Do not edit it by hand. Run scripts/sync-ggml.sh <ref> from
the repo root to reproduce or upgrade it; the script rewrites this file.
EOF
} > "${STAGE_DIR}/UPSTREAM"

# git archive preserves upstream commit times, which can be older than objects
# in an existing build tree. Refresh source mtimes so an incremental build after
# an upgrade cannot accidentally link stale ggml objects.
find "$STAGE_DIR" -type f -exec touch {} +

# ---- dry-run / check: report and stop ---------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
    REPRODUCED=0
    echo "[dry-run] would re-vendor: ${CUR_SHA:0:12} -> ${RESOLVED:0:12}"
    if [ -z "$REF_DIR" ]; then
        REPRODUCED=1
        echo "[dry-run] HEAD has no committed ggml/ — nothing to compare the recipe against."
    elif diff -rq "$REF_DIR" "$STAGE_DIR" >/tmp/sync-ggml.diff.$$ 2>/dev/null; then
        REPRODUCED=1
        echo "[dry-run] no differences — the committed ggml/ already matches this ref."
    else
        echo "[dry-run] path-level differences vs committed ggml/:"
        sed "s#${STAGE_DIR}#ggml(new)#g; s#${REF_DIR}#ggml(HEAD)#g" /tmp/sync-ggml.diff.$$ | sed 's/^/  /'
    fi
    rm -f /tmp/sync-ggml.diff.$$
    echo "[dry-run] nothing written."

    # --check turns the report into a gate. A pin bump is not a failure: asking
    # to check a DIFFERENT ref is a legitimate question ("what would change?"),
    # and only "the CURRENT pin does not reproduce the tree" is a defect.
    if [ "$CHECK" -eq 1 ]; then
        if [ "$RESOLVED" != "$CUR_SHA" ]; then
            echo "sync-ggml: --check against ${RESOLVED:0:12}, not the pinned ${CUR_SHA:0:12};"
            echo "sync-ggml: a pin bump is not a recipe failure — reporting the diff above as informational."
            exit 0
        fi
        if [ "$REPRODUCED" -eq 1 ]; then
            echo "sync-ggml: CHECK OK — recipe reproduced the committed ggml/ exactly"
            echo "sync-ggml:           (sync + patches == HEAD:ggml)."
            exit 0
        fi
        echo "sync-ggml: CHECK FAILED — the pinned ref ${CUR_SHA:0:12} does not reproduce the committed ggml/." >&2
        echo "sync-ggml: the tree has drifted from its recipe. A hand-edit must become a" >&2
        echo "sync-ggml: patch in patches/ggml/; re-run without --check to repair (or --force)." >&2
        exit 1
    fi
    exit 0
fi

# ---- swap into place --------------------------------------------------------
# Count the recipe's disagreements with the committed tree BEFORE the swap, so
# the invariant below reports what the recipe says about the tree it is
# replacing rather than about the bytes this platform happens to have on disk
# (see the REF_DIR block above). The working copy's own uncommitted edits are a
# separate question, and the guard at the top of the script already refuses to
# clobber them.
RECIPE_DIFF=0
RECIPE_REPORT=""
if [ -n "$REF_DIR" ]; then
    # `diff -rq` exits 1 when the trees differ — precisely the case this is here
    # to report — and `set -o pipefail` would abort the script on that status
    # before the invariant below could say anything: the failure branch used to
    # be unreachable, dying with a bare exit 1. `|| RC=$?` takes the status out
    # of the pipeline's way so it can be classified.
    RC=0
    RECIPE_REPORT="$(diff -rq "$REF_DIR" "$STAGE_DIR" 2>/dev/null)" || RC=$?
    [ "$RC" -le 1 ] || die "diff failed (exit ${RC}) comparing the recipe against HEAD:ggml"
    [ "$RC" -eq 0 ] || RECIPE_DIFF="$(printf '%s\n' "$RECIPE_REPORT" | sed -n '$=')"
fi

rm -rf "$GGML_DIR"
mv "$STAGE_DIR" "$GGML_DIR"

echo "sync-ggml: ggml re-vendored ${CUR_SHA:0:12} -> ${RESOLVED:0:12}"

# ---- reproducibility invariant ----------------------------------------------
# Re-applying the recipe for the pin we are already on must reproduce the
# committed tree exactly. Anything else means the tree has drifted from the
# recipe that is supposed to generate it — the silent-loss failure mode this
# vendoring scheme exists to prevent (an edit applied to the tree without a
# matching patch is one `sync-ggml.sh` away from vanishing). Fail loudly.
if [ "$RESOLVED" = "$CUR_SHA" ]; then
    if [ "$RECIPE_DIFF" = "0" ]; then
        echo "sync-ggml: recipe reproduced the committed ggml/ exactly (sync + patches == tree)."
    else
        echo "sync-ggml: ERROR — re-vendoring ${CUR_SHA:0:12} disagrees with the committed tree in ${RECIPE_DIFF} path(s)." >&2
        printf '%s\n' "$RECIPE_REPORT" \
            | sed "s#${STAGE_DIR}#ggml(new)#g; s#${REF_DIR}#ggml(HEAD)#g" \
            | sed 's/^/sync-ggml:   /' >&2
        echo "sync-ggml: the committed ggml/ does not match its own recipe (sync + patches)." >&2
        echo "sync-ggml: a hand-edit to ggml/ must become a patch in patches/ggml/, or a" >&2
        echo "sync-ggml: deliberate drop must be listed in EXCLUDES; it is otherwise one" >&2
        echo "sync-ggml: sync away from vanishing. The tree just committed above (ggml/) is" >&2
        echo "sync-ggml: the recipe output — reconcile the two and re-run this script." >&2
        if [ "$FORCE" -eq 1 ]; then
            echo "sync-ggml: --force given — keeping the regenerated tree anyway." >&2
        else
            echo "sync-ggml: ggml/ now holds the recipe output; commit it with the" >&2
            echo "sync-ggml: recipe change, or revert with 'git checkout -- ggml'." >&2
            exit 1
        fi
    fi
else
    echo "sync-ggml: ${RECIPE_DIFF} path(s) differ from the previous tree (pin bump)."
    echo "sync-ggml: review: git diff --stat -- ggml"
fi

echo "sync-ggml: next — cmake --build build --target transcribe-cli, then push to run native-ci (ggml/**)"
