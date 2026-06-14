"""Motor de simulación Monte Carlo basado en Poisson — "las simulaciones" que viste.

Idea central: si modelas cuántos goles esperas de cada equipo (lambda_home, lambda_away),
puedes SIMULAR el partido miles de veces muestreando marcadores de una distribución de
Poisson. De esas simulaciones derivas TODOS los mercados de una sola pasada:
  - 1X2 (probabilidad de victoria local / empate / visitante)
  - Over/Under (p.ej. más/menos de 2.5 goles)
  - Ambos equipos marcan (BTTS)
  - Marcador exacto más probable

Este módulo es genérico: recibe las lambdas (las estimas con una regresión de Poisson o
las derivas del Elo) y devuelve probabilidades. Es el complemento del modelo de clasificación.
"""

from __future__ import annotations

from dataclasses import dataclass

from worldcup.common import constants as C


@dataclass
class MatchProbabilities:
    home_win: float
    draw: float
    away_win: float
    over_2_5: float
    btts: float
    top_scoreline: tuple[int, int]


def simulate_match(
    lambda_home: float, lambda_away: float, n_sims: int = C.SIM_N, seed: int = 7
) -> MatchProbabilities:
    """Simula un partido `n_sims` veces y devuelve probabilidades de cada mercado."""
    import numpy as np

    rng = np.random.default_rng(seed)
    gh = rng.poisson(lambda_home, n_sims)
    ga = rng.poisson(lambda_away, n_sims)

    home_win = float((gh > ga).mean())
    draw = float((gh == ga).mean())
    away_win = float((gh < ga).mean())
    over_2_5 = float((gh + ga > 2.5).mean())
    btts = float(((gh > 0) & (ga > 0)).mean())

    # Marcador exacto más frecuente.
    from collections import Counter

    counts = Counter(zip(gh.tolist(), ga.tolist(), strict=False))
    top = max(counts, key=counts.get)

    return MatchProbabilities(home_win, draw, away_win, over_2_5, btts, top)


def goal_markets(lh: float, la: float, max_goals: int = 7) -> dict:
    """Mercados de goles EXACTOS a partir de las lambdas (sin ruido de simulación).

    Usa que la suma de dos Poisson independientes es Poisson(λ_local+λ_visita), y que cada
    marcador exacto es pmf(i,λh)·pmf(j,λa). Devuelve over/under, distribución de goles,
    marcadores más probables, portería a cero y ambos marcan.
    """
    from scipy.stats import poisson

    total = lh + la
    over = {f"over_{str(ln).replace('.', '_')}": round(float(1 - poisson.cdf(int(ln), total)), 3)
            for ln in (0.5, 1.5, 2.5, 3.5, 4.5)}

    dist, cum = {}, 0.0
    for k in range(max_goals):
        p = float(poisson.pmf(k, total))
        dist[str(k)] = round(p, 3)
        cum += p
    dist[f"{max_goals}+"] = round(max(0.0, 1 - cum), 3)

    scores = [(i, j, float(poisson.pmf(i, lh) * poisson.pmf(j, la)))
              for i in range(7) for j in range(7)]
    scores.sort(key=lambda x: -x[2])
    top = [{"marcador": f"{i}-{j}", "prob": round(p, 3)} for i, j, p in scores[:6]]

    return {
        "over_under": over,
        "distribucion_goles": dist,
        "marcadores_top": top,
        "porteria_cero_local": round(float(poisson.pmf(0, la)), 3),   # el rival no marca
        "porteria_cero_visitante": round(float(poisson.pmf(0, lh)), 3),
        "btts": round(float((1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la))), 3),
    }


def count_markets(lmbda: float, lines: list[float], max_count: int = 12) -> dict:
    """Mercados de un conteo (tarjetas, córners) modelado como Poisson(lmbda).

    Devuelve el valor esperado, la escalera over/under para las líneas dadas, la
    distribución del conteo y el valor más probable.
    """
    from scipy.stats import poisson

    over = {f"over_{str(ln).replace('.', '_')}": round(float(1 - poisson.cdf(int(ln), lmbda)), 3)
            for ln in lines}
    dist, cum = {}, 0.0
    for k in range(max_count):
        p = float(poisson.pmf(k, lmbda))
        dist[str(k)] = round(p, 3)
        cum += p
    dist[f"{max_count}+"] = round(max(0.0, 1 - cum), 3)
    pmfs = [poisson.pmf(k, lmbda) for k in range(max_count + 1)]
    return {
        "esperado": round(float(lmbda), 2),
        "over_under": over,
        "distribucion": dist,
        "mas_probable": int(max(range(len(pmfs)), key=lambda k: pmfs[k])),
    }


def elo_to_lambdas(
    home_elo: float, away_elo: float, league_avg_goals: float = C.LEAGUE_AVG_GOALS,
    neutral: bool = False,
) -> tuple[float, float]:
    """Convierte una diferencia de Elo en goles esperados (lambdas) — heurística de respaldo.

    Se usa cuando no hay historial de goles para el modelo ataque/defensa. En sede neutral
    no se aplica la ventaja de local.
    """
    home_adv = 0 if neutral else C.HOME_ADVANTAGE_ELO
    diff = (home_elo + home_adv - away_elo) / 400.0
    factor = 10 ** (diff / 2)
    return league_avg_goals * factor, league_avg_goals / factor


def lambdas_from_strength(
    home_gf, home_ga, away_gf, away_ga,
    league_avg: float = C.LEAGUE_AVG_GOALS, neutral: bool = False,
) -> tuple[float, float]:
    """Goles esperados con un modelo de ATAQUE/DEFENSA (estilo Dixon-Coles simplificado).

    λ_local = ataque_local × defensa_visita / media_liga   (y simétrico)
    donde ataque = goles que marca el equipo y defensa = goles que encaja el rival
    (promedios recientes). Aplica el factor de local salvo en sede neutral. Es más
    principista que derivarlo del Elo y alimenta la simulación Monte Carlo.
    """
    ha = home_gf if home_gf is not None else league_avg
    hd = home_ga if home_ga is not None else league_avg
    aa = away_gf if away_gf is not None else league_avg
    ad = away_ga if away_ga is not None else league_avg
    lam_home = max(0.15, ha * ad / league_avg)
    lam_away = max(0.15, aa * hd / league_avg)
    factor = 1.0 if neutral else C.HOME_GOAL_FACTOR
    return lam_home * factor, lam_away / factor
