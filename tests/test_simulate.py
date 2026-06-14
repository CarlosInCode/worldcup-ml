"""Tests de funciones puras: Elo, simulación Monte Carlo, lambdas ataque/defensa."""

from worldcup.dataplatform.features import _expected
from worldcup.predictor.simulate import (
    elo_to_lambdas,
    goal_markets,
    lambdas_from_strength,
    simulate_match,
)


def test_elo_expected_simetrico():
    assert _expected(1500, 1500) == 0.5
    assert _expected(1700, 1500) > 0.5
    assert _expected(1300, 1500) < 0.5


def test_simulacion_probabilidades_suman_uno():
    p = simulate_match(1.5, 1.0, n_sims=20000)
    assert abs(p.home_win + p.draw + p.away_win - 1.0) < 0.02
    # con más goles esperados de local, gana local más a menudo
    assert p.home_win > p.away_win


def test_neutral_quita_ventaja_local():
    # mismo equipo: en sede neutral las lambdas deben ser iguales
    lh_n, la_n = lambdas_from_strength(1.5, 1.0, 1.5, 1.0, neutral=True)
    assert abs(lh_n - la_n) < 1e-9
    # con ventaja de local, el local marca más
    lh, la = lambdas_from_strength(1.5, 1.0, 1.5, 1.0, neutral=False)
    assert lh > la


def test_lambdas_mejor_ataque_mas_goles():
    fuerte = lambdas_from_strength(2.5, 0.5, 1.0, 1.0, neutral=True)[0]
    flojo = lambdas_from_strength(0.8, 0.5, 1.0, 1.0, neutral=True)[0]
    assert fuerte > flojo


def test_elo_to_lambdas_neutral():
    lh, la = elo_to_lambdas(1600, 1600, neutral=True)
    assert abs(lh - la) < 1e-9


def test_goal_markets_coherentes():
    gm = goal_markets(1.5, 1.2)
    ou = gm["over_under"]
    # over/under decrece al subir la línea
    assert ou["over_0_5"] > ou["over_1_5"] > ou["over_2_5"] > ou["over_3_5"] > ou["over_4_5"]
    # la distribución de goles suma ~1 y las probabilidades son válidas
    assert abs(sum(gm["distribucion_goles"].values()) - 1.0) < 0.01
    assert 0 <= gm["btts"] <= 1
    assert len(gm["marcadores_top"]) == 6
