#!/usr/bin/env bash
set -euo pipefail

PGMAJOR="${1:?usage: build-rpm.sh <pg-major> <version> <staged-tree-dir> <output-dir>}"
VERSION="${2:?version required}"
STAGE="${3:?staged tree dir required}"
OUTDIR="${4:?output dir required}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$(cd "$STAGE" && pwd)"

SO="$STAGE/usr/pgsql-$PGMAJOR/lib/pg_fela.so"
CONTROL="$STAGE/usr/pgsql-$PGMAJOR/share/extension/pg_fela.control"
test -f "$SO" || { echo "FAIL: $SO not staged - was cargo pgrx package run against PGDG yum postgresql${PGMAJOR}-devel?" >&2; exit 1; }
test -f "$CONTROL" || { echo "FAIL: $CONTROL not staged" >&2; exit 1; }

TOPDIR="$(mktemp -d)"
mkdir -p "$TOPDIR"/{BUILD,RPMS,SRPMS,SPECS,SOURCES} "$OUTDIR"

rpmbuild -bb \
  --define "_topdir $TOPDIR" \
  --define "pgmajor $PGMAJOR" \
  --define "version $VERSION" \
  --buildroot "$STAGE" \
  "$HERE/pg_fela.spec"

find "$TOPDIR/RPMS" -name '*.rpm' -exec cp -v {} "$OUTDIR/" \;
rm -rf "$TOPDIR"
