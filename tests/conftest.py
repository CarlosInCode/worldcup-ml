"""Fixtures de pytest: aíslan cada test en un DATA_DIR temporal (sin tocar tus datos reales)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from worldcup.common.config import settings


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Apunta el proyecto a un directorio temporal (warehouse, models, mlflow.db)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    (tmp_path / "data").mkdir()
    return tmp_path


def make_matches(n_teams=12, n_matches=400):
    """Genera partidos sintéticos coherentes para tests de features/modelos."""
    base = datetime(2022, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n_matches):
        h = (i % n_teams) + 1
        a = ((i * 5) % n_teams) + 1
        if h == a:
            a = (a % n_teams) + 1
        hg, ag = i % 4, (i + 1) % 3
        res = 0 if hg > ag else 1 if hg == ag else 2
        rows.append({
            "match_id": i, "kickoff_utc": base + timedelta(days=i),
            "league_id": 1, "season": 2022,
            "home_team_id": h, "away_team_id": a,
            "home_team": f"T{h}", "away_team": f"T{a}",
            "home_goals": hg, "away_goals": ag, "status": "FT",
            "referee": f"Ref{i % 6}", "result_1x2": res,
        })
    return pl.DataFrame(rows)


def seed_warehouse(matches: pl.DataFrame, league_meta: pl.DataFrame | None = None):
    """Escribe tablas Silver sintéticas en el warehouse temporal."""
    from worldcup.common.db import connect

    with connect() as con:
        con.register("m", matches.to_arrow())
        con.execute("CREATE OR REPLACE TABLE matches AS SELECT * FROM m")
        if league_meta is not None:
            con.register("lm", league_meta.to_arrow())
            con.execute("CREATE OR REPLACE TABLE league_meta AS SELECT * FROM lm")
