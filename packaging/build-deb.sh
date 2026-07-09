#!/usr/bin/env bash
set -euo pipefail

PGMAJOR="${1:?usage: build-deb.sh <pg-major> <version> <staged-tree-dir> <output.deb> [arch]}"
VERSION="${2:?version required}"
STAGE="${3:?staged tree dir required}"
OUT="${4:?output .deb path required}"
ARCH="${5:-amd64}"

test -d "$STAGE" || { echo "FAIL: staged tree not found: $STAGE" >&2; exit 1; }

SO="$STAGE/usr/lib/postgresql/$PGMAJOR/lib/pg_fela.so"
CONTROL="$STAGE/usr/share/postgresql/$PGMAJOR/extension/pg_fela.control"
test -f "$SO" || { echo "FAIL: $SO not staged - was cargo pgrx package run against PGDG apt PostgreSQL $PGMAJOR?" >&2; exit 1; }
test -f "$CONTROL" || { echo "FAIL: $CONTROL not staged" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cp -a "$STAGE/." "$WORK/"
rm -rf "$WORK/DEBIAN"
mkdir -p "$WORK/DEBIAN"

SIZE_KB="$(du -sk --exclude=DEBIAN "$WORK" | cut -f1)"
PKG="pg-fela-pg${PGMAJOR}"

cat > "$WORK/DEBIAN/control" <<EOF
Package: $PKG
Version: $VERSION
Section: database
Priority: optional
Architecture: $ARCH
Depends: postgresql-$PGMAJOR
Maintainer: Lowdown Labs <dev@gimmelowdown.com>
Homepage: https://github.com/Lowdown-Labs/pg_fela
Installed-Size: $SIZE_KB
Description: In-situ AutoML for PostgreSQL $PGMAJOR (FelaTab model)
 pg_fela adds fela_automl() and related SQL functions that run zero-config
 AutoML predictions directly inside PostgreSQL $PGMAJOR. The FelaTab model is
 embedded in the extension binary at build time; no GUC and no separate data
 file are needed before use.
EOF

mkdir -p "$(dirname "$OUT")"
dpkg-deb --build --root-owner-group "$WORK" "$OUT"
echo "built: $OUT"
