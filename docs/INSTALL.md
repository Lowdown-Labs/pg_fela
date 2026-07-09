# Installing pg_fela

The FelaTab model is embedded in the extension binary (`pg_fela.so`) at build time, so at run time
there is no GUC to set, no file to place, and no network access before `SELECT * FROM
fela_automl(...)` works. The weights are NOT committed to git or Git-LFS: `build.rs` fetches them at
COMPILE time from `FELATAB_MODEL_URL` (defaults to the Lowdown CDN) and `include_bytes!`s them into
the binary, after verifying `FELATAB_MODEL_SHA256`. For an offline build, set `FELATAB_WEIGHTS` to a
local `felatab_int8.safetensors` and build.rs copies that instead of downloading. `SET
fela.model_path` at run time still loads a different model from a filesystem path.

Because the model is embedded, source builds DO need the weights available at build time (fetched or
`FELATAB_WEIGHTS`). Prebuilt images/packages already have them embedded.

Three install paths, in order of how little you need to do:

1. [Docker (recommended)](#1-docker-recommended): zero config, one command.
2. [Prebuilt package from GitHub Releases](#2-prebuilt-package-from-github-releases): `.deb`,
   `.rpm`, or a plain tarball, per supported PostgreSQL major.
3. [Build from source](#3-build-from-source-cargo-pgrx) with `cargo-pgrx`.

macOS (Docker Desktop / Apple Silicon) users: see [macOS](#macos-docker-desktop--apple-silicon).

> Note on managed services: AWS RDS (and similar managed PostgreSQL) only load extensions on their
> own allowlist. Embedding the model does not change that - pg_fela is not on the RDS allowlist, so
> it will not load on stock RDS. Use Docker, a self-managed host, or another managed provider that
> permits custom extensions.

---

## 1. Docker (recommended)

Prebuilt multi arch (`linux/amd64` + `linux/arm64`) images, one per supported PostgreSQL major:

```bash
docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d ghcr.io/lowdown-labs/pg_fela:pg18
# also published: :pg16, :pg17, :pg18  (:latest == :pg18)
```

The extension is auto created in the bootstrap database on first boot
(`docker-entrypoint-initdb.d/00-create-extension.sql`). A fresh container is ready immediately:

```bash
psql -h localhost -U postgres -c \
  "CREATE TABLE t (x1 float8, x2 float8, y text);
   INSERT INTO t VALUES (1,1,'a'),(1.2,0.9,'a'),(9,9,'b'),(9.1,8.9,'b'),(5,5,NULL);
   SELECT * FROM fela_automl('t','y');"
```

No GUC and no query-time network access are needed: the model is embedded in `pg_fela.so`, and
everything else the query touches is in the binary too.

**Building it yourself** (same Dockerfile the release pipeline uses):

```bash
docker build --build-arg PG_MAJOR=18 -t pg_fela:local .
```

`PG_MAJOR` accepts `16`, `17`, or `18`. The build's `build.rs` downloads the weights from
`FELATAB_MODEL_URL` (defaults to the Lowdown CDN), verifies `FELATAB_MODEL_SHA256`, and embeds them
into `pg_fela.so`. To self-host, pass
`--build-arg FELATAB_MODEL_URL=... --build-arg FELATAB_MODEL_SHA256=...`.

---

## macOS (Docker Desktop / Apple Silicon)

The published images are multi-arch, so an Apple-Silicon Mac with Docker Desktop pulls and runs the
native `linux/arm64` variant automatically:

```bash
docker run --rm -e POSTGRES_PASSWORD=postgres -p 5432:5432 ghcr.io/lowdown-labs/pg_fela:pg18
# in another terminal:
psql -h localhost -U postgres -c "CREATE EXTENSION pg_fela;" \
  -c "CREATE TABLE t (x1 float8, x2 float8, y text);
      INSERT INTO t VALUES (1,1,'a'),(1.2,0.9,'a'),(9,9,'b'),(9.1,8.9,'b'),(5,5,NULL);
      SELECT * FROM fela_automl('t','y');"
```

Self-build the arm64 image locally (embeds the model at build time, same as above):

```bash
docker buildx build --platform linux/arm64 --build-arg PG_MAJOR=18 -t pg_fela:arm64 --load .
docker run --rm -e POSTGRES_PASSWORD=postgres -p 5432:5432 pg_fela:arm64
```

---

## 2. Prebuilt package from GitHub Releases

Every tagged release attaches, per supported PostgreSQL major (`pg16` / `pg17` / `pg18`, `amd64`):

- a `.deb` package (`pg-fela-pgNN_<version>_amd64.deb`), built against the `apt.postgresql.org`
  (PGDG apt) layout
- an `.rpm` package (`pg-fela-pgNN-<version>-1.el9.x86_64.rpm`), built against the PGDG yum
  layout (`/usr/pgsql-NN/...`)
- a plain `pg_fela-pgNN-amd64.tar.gz` tarball, the exact tree `cargo pgrx package` produces,
  rooted where your PostgreSQL install expects it

Grab the asset for your distro and PostgreSQL major from the
[GitHub Releases page](https://github.com/Lowdown-Labs/pg_fela/releases), then install it with your
platform's normal package tool.

### Debian / Ubuntu (`.deb`)

```bash
# Download pg-fela-pg18_1.0.0_amd64.deb from the Releases page, then:
sudo dpkg -i pg-fela-pg18_1.0.0_amd64.deb
psql -c "CREATE EXTENSION pg_fela;"
```

### RHEL / Rocky / AlmaLinux with the PGDG yum repo (`.rpm`)

Built against the PGDG yum layout (`postgresqlNN-server` from
[download.postgresql.org's yum repo](https://www.postgresql.org/download/linux/redhat/), which
installs to `/usr/pgsql-NN/...`, not the `/usr/lib/postgresql/NN` apt layout above):

```bash
# Download pg-fela-pg18-1.0.0-1.el9.x86_64.rpm from the Releases page, then:
sudo rpm -i pg-fela-pg18-1.0.0-1.el9.x86_64.rpm
psql -c "CREATE EXTENSION pg_fela;"
```

### Plain tarball (any distro)

```bash
# 1. Download + inspect the tarball for your PG major, e.g.:
tar tzf pg_fela-pg18-amd64.tar.gz | head
#   usr/lib/postgresql/18/lib/pg_fela.so
#   usr/share/postgresql/18/extension/pg_fela.control
#   usr/share/postgresql/18/extension/pg_fela--1.0.0.sql

# 2. Extract onto your PostgreSQL host, rooted at `/` (matches pg_config --pkglibdir/--sharedir
#    for a stock apt.postgresql.org install; adjust the paths if yours differs):
sudo tar xzf pg_fela-pg18-amd64.tar.gz -C /

# 3. Enable it per database:
psql -c "CREATE EXTENSION pg_fela;"
```

Each of these embeds the model into `pg_fela.so` at build time, so there is no separate data file to
ship and a fresh `CREATE EXTENSION` + a prediction work with no GUC. These are the same artifacts
`test/run_pgrx_test.sh` builds and installs in CI, so what you download is what CI already proved
installs cleanly and passes the honesty gate.

---

## 3. Build from source (`cargo-pgrx`)

Needs a Rust toolchain, `cargo-pgrx`, and PostgreSQL dev headers for the target major (see
`.github/workflows/ci.yml` for the exact apt/pgdg setup this project uses).

```bash
cargo install cargo-pgrx --version 0.19.1 --locked
cargo pgrx init --pg18 /usr/lib/postgresql/18/bin/pg_config
cargo pgrx install --pg-config /usr/lib/postgresql/18/bin/pg_config --release
psql -c "CREATE EXTENSION pg_fela;"
```

`build.rs` needs the weights at build time. By default it downloads them from `FELATAB_MODEL_URL`
(the Lowdown CDN) and verifies `FELATAB_MODEL_SHA256`. For an offline build, point `FELATAB_WEIGHTS`
at a local `felatab_int8.safetensors` and build.rs copies it (still sha-checked):

```bash
FELATAB_WEIGHTS=/path/to/felatab_int8.safetensors \
  cargo pgrx install --pg-config /usr/lib/postgresql/18/bin/pg_config --release
```

Once built, the model is embedded; running a prediction needs no model on disk and no GUC.

To build a redistributable tarball instead of installing directly:

```bash
cargo pgrx package --pg-config /usr/lib/postgresql/18/bin/pg_config
tar czf pg_fela-pg18-amd64.tar.gz -C target/release pg_fela-pg18
```

Swap `18` for `16` or `17` throughout to target a different supported PostgreSQL major.
