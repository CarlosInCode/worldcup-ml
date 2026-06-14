"""Acceso a DuckDB: nuestro warehouse analítico embebido (capas Silver y Gold).

DuckDB es como "SQLite para analítica": un solo archivo, cero servidor, lee Parquet
nativo y es rapidísimo. Cuando escales a varios usuarios o TBs, migras a Postgres/
BigQuery cambiando solo esta capa.
"""

from __future__ import annotations

from contextlib import contextmanager

import duckdb

from worldcup.common.config import settings


@contextmanager
def connect():
    """Context manager que devuelve una conexión a DuckDB y la cierra al final.

        with connect() as con:
            df = con.sql("SELECT * FROM matches LIMIT 5").pl()
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(settings.duckdb_path))
    try:
        yield con
    finally:
        con.close()


def read_parquet_glob(con: duckdb.DuckDBPyConnection, glob: str):
    """Lee múltiples Parquet (p.ej. 'data/bronze/fixtures/**/*.parquet') como una relación."""
    return con.sql(f"SELECT * FROM read_parquet('{glob}', union_by_name=true)")
