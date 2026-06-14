"""Tests de modelos: clasificación de competición, Poisson de conteo y registro."""

import polars as pl

from tests.conftest import seed_warehouse
from worldcup.common.db import connect
from worldcup.dataplatform.silver import _classify_competition


def test_clasificacion_competicion():
    assert _classify_competition("World Cup") == ("tournament", True)
    assert _classify_competition("World Cup - Qualification Europe") == ("qualifier", False)
    assert _classify_competition("UEFA Nations League") == ("nations", False)
    assert _classify_competition("Friendlies") == ("friendly", False)
    assert _classify_competition("Premier League") == ("league", False)


def test_poisson_supera_baseline(workspace):
    # card_features sintético donde las tarjetas dependen del árbitro (señal real)
    rows = []
    for i in range(400):
        rb = float(2 + i % 5)
        rows.append({"match_id": i, "kickoff_utc": f"2022-01-{(i % 27) + 1:02d}T12:00:00",
                     "ref_avg_cards": rb, "home_avg_cards": 3.0, "away_avg_cards": 3.0,
                     "total_cards": int(rb + 1.0 - (i % 3))})
    with connect() as con:
        con.register("cf", pl.DataFrame(rows).to_arrow())
        con.execute("CREATE OR REPLACE TABLE card_features AS SELECT * FROM cf")
    from worldcup.predictor.cards import train_cards

    m = train_cards()
    assert m["mae"] < m["baseline_mae_media"]  # el modelo debe aportar


def test_registry_champion_challenger(workspace):
    from worldcup.predictor import registry

    saved = []
    # primer modelo: no hay campeón -> promociona
    d1 = registry.promote("x", 1.0, lambda: saved.append("a"), higher_is_better=False)
    assert d1["promoted"] and saved == ["a"]
    # peor (mayor log-loss) -> NO promociona
    d2 = registry.promote("x", 2.0, lambda: saved.append("b"), higher_is_better=False)
    assert not d2["promoted"] and saved == ["a"]
    # mejor -> promociona
    d3 = registry.promote("x", 0.5, lambda: saved.append("c"), higher_is_better=False)
    assert d3["promoted"] and saved == ["a", "c"]
