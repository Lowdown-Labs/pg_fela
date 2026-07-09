%global debug_package %{nil}
%global __os_install_post %{nil}
%global _build_id_links none

Name:           pg-fela-pg%{pgmajor}
Version:        %{version}
Release:        1%{?dist}
Summary:        In-situ AutoML for PostgreSQL %{pgmajor} (FelaTab model)

License:        MIT
URL:            https://github.com/Lowdown-Labs/fela_pg
Group:          Applications/Databases
BuildArch:      x86_64
Requires:       postgresql%{pgmajor}-server
AutoReqProv:    no

%description
pg_fela adds fela_automl() and related SQL functions that run zero-config AutoML
predictions directly inside PostgreSQL %{pgmajor}. The FelaTab model is embedded in the
extension binary at build time; no GUC and no separate data file are needed before use.

%files
/usr/pgsql-%{pgmajor}/lib/pg_fela.so
/usr/pgsql-%{pgmajor}/share/extension/pg_fela.control
/usr/pgsql-%{pgmajor}/share/extension/pg_fela--%{version}.sql
