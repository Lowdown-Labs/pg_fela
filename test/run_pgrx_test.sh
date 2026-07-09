#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PG_CONFIG="${PG_CONFIG:-$(command -v pg_config)}"
[ -x "$PG_CONFIG" ] || { echo "FAIL: pg_config not found (set PG_CONFIG=/path/to/pg_config)"; exit 1; }
PGMAJOR="$("$PG_CONFIG" --version | sed -E 's/[^0-9]*([0-9]+).*/\1/')"
PGTAG="pg${PGMAJOR}"

GOLDEN="$REPO/tests/fixtures/felatab/golden.json"
test -s "$GOLDEN" || { echo "FAIL: missing fixture $GOLDEN"; exit 1; }
export PATH="$HOME/.cargo/bin:$PATH"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/pg_fela_pgrx.XXXXXX")"
PGDATA="$WORK/pgdata"
PGSOCK="$WORK/sock"
PGPORT="${PGPORT:-54331}"
mkdir -p "$PGSOCK"

cleanup() {
    if [ -f "$PGDATA/postmaster.pid" ]; then
        pg_ctl -D "$PGDATA" -m immediate stop >/dev/null 2>&1 || true
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

echo "== [1/5] cargo pgrx package (stock $PGTAG staging tree) =="
( cd "$REPO" && cargo pgrx package --pg-config "$PG_CONFIG" >/dev/null )
STAGE="$REPO/target/release/pg_fela-$PGTAG"
SHAREDIR="$("$PG_CONFIG" --sharedir)"
PKGLIB="$("$PG_CONFIG" --pkglibdir)"
CTRL_DIR="$STAGE$SHAREDIR"
LIB_DIR="$STAGE$PKGLIB"
test -f "$CTRL_DIR/extension/pg_fela.control" || { echo "FAIL: control not staged ($CTRL_DIR)"; exit 1; }
test -f "$LIB_DIR/pg_fela.so" || { echo "FAIL: .so not staged ($LIB_DIR)"; exit 1; }
echo "   staged: $LIB_DIR/pg_fela.so"

echo "== [2/5] throwaway PostgreSQL $PGMAJOR cluster =="
initdb -D "$PGDATA" -U postgres --no-sync >/dev/null 2>&1
pg_ctl -D "$PGDATA" \
  -o "-k $PGSOCK -p $PGPORT -c listen_addresses='' -c extension_control_path=$CTRL_DIR -c dynamic_library_path=$LIB_DIR" \
  -w start >/dev/null
PSQL="psql -h $PGSOCK -p $PGPORT -U postgres -v ON_ERROR_STOP=1 -qAt"

echo "== [3/5] CREATE EXTENSION + honesty gate (vs committed golden.json) =="
$PSQL -c "CREATE EXTENSION pg_fela;" >/dev/null
echo "   version: $($PSQL -c "SELECT fela_version();")"

SQL_FIXTURE="$WORK/fixture.sql"
python3 - "$GOLDEN" > "$SQL_FIXTURE" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))["cls"]
nf, ns, nq, ncls = g["n_feat"], g["n_support"], g["n_query"], g["ncls"]
x = g["x"]
support = x[: ns * nf]
query = x[ns * nf : ns * nf + nq * nf]
labels = [int(v) for v in g["y_support"]]
fa = lambda v: "ARRAY[" + ",".join(f"{z:.9f}" for z in v) + "]::float8[]"
ia = lambda v: "ARRAY[" + ",".join(str(z) for z in v) + "]::int[]"
print(f"SELECT array_to_string(fela_classify({fa(query)}, {fa(support)}, {ia(labels)}, {nf}, {ncls}), ',');")
PY
GOT="$($PSQL -f "$SQL_FIXTURE" | tail -1)"
python3 - "$GOLDEN" "$GOT" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))
ref, tol = g["cls"]["probs"], g["tol"]
got = [float(x) for x in sys.argv[2].split(",")]
assert len(got) == len(ref), f"len mismatch {len(got)} vs {len(ref)}"
maxerr = max(abs(a - b) for a, b in zip(got, ref))
print(f"   HONESTY GATE: max|pgrx - golden| = {maxerr:.3e}  (tol {tol})")
assert maxerr < tol, f"MISMATCH: {maxerr} >= {tol}"
print("   HONESTY GATE: PASS  (in-DB fela_classify == frozen fp32 reference within int8 parity bar)")
PY

echo "== [4/5] AutoML surface on an iris-like table (examples/automl_showcase.sql) =="
export PGHOST="$PGSOCK" PGPORT="$PGPORT" PGUSER="postgres"
$PSQL -f "$REPO/examples/automl_showcase.sql"

echo "== [5/5] latency (real, measured) =="
$PSQL <<SQL
SELECT fela_classify(ARRAY[5,3.4,1.5]::float8[],
  ARRAY[5.1,3.5,1.4, 4.9,3.0,1.4, 7.0,3.2,4.7, 6.4,3.2,4.5, 6.3,3.3,6.0, 5.8,2.7,5.1]::float8[],
  ARRAY[0,0,1,1,2,2]::int[], 3, 3) IS NOT NULL AS warmed;
\timing on
SELECT count(*) FROM generate_series(1,100) g,
  LATERAL fela_classify(ARRAY[5,3.4,1.5]::float8[],
    ARRAY[5.1,3.5,1.4, 4.9,3.0,1.4, 7.0,3.2,4.7, 6.4,3.2,4.5, 6.3,3.3,6.0, 5.8,2.7,5.1]::float8[],
    ARRAY[0,0,1,1,2,2]::int[], 3, 3) p;
\timing off
SQL

echo ""
echo "ALL STEPS PASSED ($PGTAG)."
