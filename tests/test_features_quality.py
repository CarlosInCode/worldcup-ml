"""Tests de la capa Gold (anti-leakage, sede neutral) y de las puertas de calidad."""

import polars as pl

from tests.conftest import make_matches, seed_warehouse
from worldcup.common.db import connect


def test_features_anti_leakage_y_columnas(workspace):
    seed_warehouse(make_matches())
    from worldcup.dataplatform.features import build_match_features

    n = build_match_features()
    assert n > 0
    with connect() as con:
        df = con.sql("SELECT * FROM match_features ORDER BY kickoff_utc").pl()
    # El PRIMER partido cronológico no puede tener historial -> Elo base (anti-leakage).
    assert df["home_elo"][0] == 1500.0
    assert df["away_elo"][0] == 1500.0
    # Columnas nuevas presentes
    for col in ["is_neutral", "importance", "home_sos", "away_sos"]:
        assert col in df.columns


def test_sede_neutral_marca_feature(workspace):
    matches = make_matches(n_matches=300)
    meta = pl.DataFrame([{"league_id": 1, "league_name": "World Cup",
                          "competition_type": "tournament", "is_neutral": True}])
    seed_warehouse(matches, meta)
    from worldcup.dataplatform.features import build_match_features

    build_match_features()
    with connect() as con:
        vals = con.sql("SELECT DISTINCT is_neutral FROM match_features").pl()["is_neutral"].to_list()
    assert vals == [1]  # todos marcados como neutral


def test_quality_gate_detecta_basura(workspace):
    seed_warehouse(make_matches())
    from worldcup import quality

    assert quality.gate(quality.run_checks()) is True
    # inyecta un resultado inválido -> la puerta crítica debe cerrarse
    with connect() as con:
        con.execute("INSERT INTO matches (match_id, result_1x2) VALUES (999999, 7)")
    assert quality.gate(quality.run_checks()) is False
