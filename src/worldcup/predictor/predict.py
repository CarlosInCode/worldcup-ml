"""Predicción on-demand para un partido concreto.

Combina los dos enfoques:
  1. El modelo XGBoost entrenado (probabilidades 1X2 aprendidas de los datos).
  2. El motor de simulación Poisson (deriva over/under, BTTS, marcador exacto).

Calcula los features (Elo, forma reciente, H2H) al vuelo desde la tabla `matches`,
usando las MISMAS definiciones que el entrenamiento, para que el vector sea coherente.
"""

from __future__ import annotations

from pathlib import Path

from worldcup.common.db import connect
from worldcup.predictor.simulate import (
    count_markets,
    elo_to_lambdas,
    goal_markets,
    lambdas_from_strength,
    simulate_match,
)

_MODEL_PATH = Path("models/model_1x2.joblib")
_CARDS_MODEL = Path("models/model_cards.pkl")
_CORNERS_MODEL = Path("models/model_corners.pkl")
_PLAYER_SHOTS_MODEL = Path("models/model_player_shots.pkl")


def _poisson_over(lmbda: float, line: float) -> float:
    """P(conteo > line) para una Poisson de media lmbda (p.ej. over 3.5 tarjetas)."""
    from scipy.stats import poisson

    return float(1.0 - poisson.cdf(int(line), lmbda))


def _team_recent_cards(con, team_id: int, n: int = 5):
    rows = con.sql(
        f"""
        SELECT s.tc FROM (
            SELECT match_id, sum(coalesce(yellow_cards,0)) + sum(coalesce(red_cards,0)) AS tc
            FROM match_statistics GROUP BY match_id
        ) s JOIN matches m ON m.match_id = s.match_id
        WHERE m.home_team_id = {team_id} OR m.away_team_id = {team_id}
        ORDER BY m.kickoff_utc DESC LIMIT {n}
        """
    ).fetchall()
    return sum(r[0] for r in rows) / len(rows) if rows else None


def _referee_avg_cards(con, referee: str | None):
    if not referee:
        return None
    row = con.sql(
        """
        SELECT avg(s.tc) FROM (
            SELECT match_id, sum(coalesce(yellow_cards,0)) + sum(coalesce(red_cards,0)) AS tc
            FROM match_statistics GROUP BY match_id
        ) s JOIN matches m ON m.match_id = s.match_id
        WHERE m.referee = ?
        """,
        params=[referee],
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _team_recent_corners(con, team_id: int, n: int = 5):
    """(córners a favor, en contra) promedio reciente del equipo."""
    rows = con.sql(
        f"""
        SELECT CASE WHEN m.home_team_id = {team_id} THEN hc.corner_kicks ELSE ac.corner_kicks END AS cf,
               CASE WHEN m.home_team_id = {team_id} THEN ac.corner_kicks ELSE hc.corner_kicks END AS ca
        FROM matches m
        JOIN match_statistics hc ON hc.match_id = m.match_id AND hc.team_id = m.home_team_id
        JOIN match_statistics ac ON ac.match_id = m.match_id AND ac.team_id = m.away_team_id
        WHERE (m.home_team_id = {team_id} OR m.away_team_id = {team_id})
          AND hc.corner_kicks IS NOT NULL AND ac.corner_kicks IS NOT NULL
        ORDER BY m.kickoff_utc DESC LIMIT {n}
        """
    ).fetchall()
    if not rows:
        return None, None
    return sum(r[0] for r in rows) / len(rows), sum(r[1] for r in rows) / len(rows)


def _team_recent_shots(con, team_id: int, n: int = 10):
    """Promedio de tiros totales recientes del equipo (volumen ofensivo)."""
    rows = con.sql(
        f"""
        SELECT CASE WHEN m.home_team_id = {team_id} THEN hc.total_shots ELSE ac.total_shots END AS sh
        FROM matches m
        JOIN match_statistics hc ON hc.match_id = m.match_id AND hc.team_id = m.home_team_id
        JOIN match_statistics ac ON ac.match_id = m.match_id AND ac.team_id = m.away_team_id
        WHERE (m.home_team_id = {team_id} OR m.away_team_id = {team_id})
          AND hc.total_shots IS NOT NULL AND ac.total_shots IS NOT NULL
        ORDER BY m.kickoff_utc DESC LIMIT {n}
        """
    ).fetchall()
    return sum(r[0] for r in rows) / len(rows) if rows else None


def _glm_lambda(model, values: list[float]) -> float:
    import numpy as np

    X = np.array([[1.0, *values]])  # el modelo se entrenó con constante delante
    return float(model.predict(X)[0])


def _latest_elo(con, team_id: int) -> float:
    row = con.sql(
        f"""
        SELECT elo FROM (
            SELECT home_elo AS elo, kickoff_utc FROM match_features WHERE home_team_id = {team_id}
            UNION ALL
            SELECT away_elo AS elo, kickoff_utc FROM match_features WHERE away_team_id = {team_id}
        ) ORDER BY kickoff_utc DESC LIMIT 1
        """
    ).fetchone()
    return float(row[0]) if row else 1500.0


def _recent_form(con, team_id: int, n: int = 5):
    """(pts_avg, gf_avg, ga_avg) de los últimos n partidos del equipo. None si no hay."""
    rows = con.sql(
        f"""
        SELECT home_team_id, home_goals, away_goals, result_1x2
        FROM matches
        WHERE (home_team_id = {team_id} OR away_team_id = {team_id}) AND home_goals IS NOT NULL
        ORDER BY kickoff_utc DESC LIMIT {n}
        """
    ).fetchall()
    if not rows:
        return None, None, None
    pts = gf = ga = 0
    for home_id, hg, ag, res in rows:
        if home_id == team_id:
            gf += hg
            ga += ag
            pts += 3 if res == 0 else 1 if res == 1 else 0
        else:
            gf += ag
            ga += hg
            pts += 3 if res == 2 else 1 if res == 1 else 0
    k = len(rows)
    return pts / k, gf / k, ga / k


def _h2h_ppg(con, home_id: int, away_id: int):
    rows = con.sql(
        f"""
        SELECT home_team_id, result_1x2 FROM matches
        WHERE ((home_team_id={home_id} AND away_team_id={away_id})
            OR (home_team_id={away_id} AND away_team_id={home_id}))
          AND home_goals IS NOT NULL
        """
    ).fetchall()
    if not rows:
        return None
    pts = 0
    for hist_home, res in rows:
        if res == 1:
            pts += 1
        elif res == 0:
            pts += 3 if hist_home == home_id else 0
        else:
            pts += 0 if hist_home == home_id else 3
    return pts / len(rows)


def _team_sos(con, team_id: int, n: int = 5):
    """Strength-of-schedule: Elo medio de los rivales enfrentados recientemente."""
    rows = con.sql(
        f"""
        SELECT CASE WHEN home_team_id = {team_id} THEN away_elo ELSE home_elo END AS opp_elo
        FROM match_features
        WHERE home_team_id = {team_id} OR away_team_id = {team_id}
        ORDER BY kickoff_utc DESC LIMIT {n}
        """
    ).fetchall()
    return sum(r[0] for r in rows) / len(rows) if rows else None


def _player_recent(con, player_id: int, n: int = 5):
    """(shots_on_avg, shots_total_avg, minutes_avg, nombre) recientes del jugador."""
    rows = con.sql(
        f"""
        SELECT p.shots_on, p.shots_total, p.minutes, p.player_name
        FROM player_match_stats p
        JOIN matches m ON m.match_id = p.match_id
        WHERE p.player_id = {player_id} AND p.shots_on IS NOT NULL
        ORDER BY m.kickoff_utc DESC LIMIT {n}
        """
    ).fetchall()
    if not rows:
        return None, None, None, None
    k = len(rows)
    son = sum(r[0] or 0 for r in rows) / k
    sht = sum(r[1] or 0 for r in rows) / k
    mins = sum(r[2] or 0 for r in rows) / k
    return son, sht, mins, rows[0][3]


def _team_top_shooters(con, team_id: int, n: int = 5):
    """Principales rematadores de un equipo según sus últimos ~12 partidos (squad actual)."""
    return con.sql(
        f"""
        WITH team_matches AS (
            SELECT match_id FROM matches
            WHERE home_team_id = {team_id} OR away_team_id = {team_id}
            ORDER BY kickoff_utc DESC LIMIT 12
        )
        SELECT p.player_id, any_value(p.player_name) AS name, avg(p.shots_on) AS son
        FROM player_match_stats p
        JOIN team_matches tm ON tm.match_id = p.match_id
        WHERE p.team_id = {team_id} AND p.player_id != 0 AND p.shots_on IS NOT NULL
        GROUP BY p.player_id
        ORDER BY son DESC NULLS LAST
        LIMIT {n}
        """
    ).fetchall()


def _predict_team_shots(con, model, team_id: int, is_home: bool, n: int = 4) -> list[dict]:
    out = []
    for pid, name, _ in _team_top_shooters(con, team_id, n):
        son, sht, mins, _nm = _player_recent(con, pid)
        if son is None:
            continue
        lam = _glm_lambda(model, [son, sht, mins, 1.0 if is_home else 0.0])
        out.append({
            "player": name,
            "shots_on_esperados": round(lam, 2),
            "prob_1_o_mas": round(_poisson_over(lam, 0.5), 3),
        })
    out.sort(key=lambda x: -x["shots_on_esperados"])
    return out


def predict_player_shots(player_id: int, is_home: bool = True) -> dict:
    """Predice los tiros a puerta de un jugador en su próximo partido (Poisson).

    Requiere el modelo entrenado (`wc train-player-shots`) y datos del endpoint `players`.
    """
    if not _PLAYER_SHOTS_MODEL.exists():
        return {"error": "Modelo no entrenado. Corre `wc train-player-shots` "
                         "(necesita el endpoint players ingerido)."}
    with connect() as con:
        son, sht, mins, name = _player_recent(con, player_id)
    if son is None:
        return {"error": f"Sin historial del jugador {player_id} en player_match_stats."}

    from statsmodels.iolib.smpickle import load_pickle

    model = load_pickle(str(_PLAYER_SHOTS_MODEL))
    # orden player_shots._FEATURES: [shots_on_avg, shots_total_avg, minutes_avg, is_home]
    lam = _glm_lambda(model, [son, sht, mins, 1.0 if is_home else 0.0])
    return {
        "player_id": player_id,
        "player_name": name,
        "shots_on_target_esperados": round(lam, 2),
        "prob_1_o_mas": round(_poisson_over(lam, 0.5), 3),
        "prob_2_o_mas": round(_poisson_over(lam, 1.5), 3),
        "prob_3_o_mas": round(_poisson_over(lam, 2.5), 3),
        "promedio_reciente_tiros_a_puerta": round(son, 2),
    }


def predict_match(home_team_id: int, away_team_id: int, referee: str | None = None,
                  neutral: bool = False) -> dict:
    """Predice un partido. Devuelve 1X2, goles (over/under, BTTS, marcador), tarjetas y córners.

    `referee`: nombre del árbitro (opcional, mejora las tarjetas).
    `neutral`: True para sede neutral (Mundial) -> sin ventaja de local.
    """
    with connect() as con:
        home_elo = _latest_elo(con, home_team_id)
        away_elo = _latest_elo(con, away_team_id)
        hp, hgf, hga = _recent_form(con, home_team_id)
        ap, agf, aga = _recent_form(con, away_team_id)
        h2h = _h2h_ppg(con, home_team_id, away_team_id)
        home_sos = _team_sos(con, home_team_id)
        away_sos = _team_sos(con, away_team_id)
        # Datos para tarjetas/córners (pueden ser None si aún no bajaste el detalle).
        ref_cards = _referee_avg_cards(con, referee)
        home_cards = _team_recent_cards(con, home_team_id)
        away_cards = _team_recent_cards(con, away_team_id)
        home_cf, home_ca = _team_recent_corners(con, home_team_id)
        away_cf, away_ca = _team_recent_corners(con, away_team_id)
        home_shots = _team_recent_shots(con, home_team_id)
        away_shots = _team_recent_shots(con, away_team_id)

    # Goles esperados con el modelo ataque/defensa (cae al Elo si no hay historial).
    if None not in (hgf, hga, agf, aga):
        lh, la = lambdas_from_strength(hgf, hga, agf, aga, neutral=neutral)
    else:
        lh, la = elo_to_lambdas(home_elo, away_elo, neutral=neutral)
    sim = simulate_match(lh, la)
    result = {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_elo": round(home_elo, 1),
        "away_elo": round(away_elo, 1),
        "sede_neutral": neutral,
        "goles_esperados": {"local": round(lh, 2), "visitante": round(la, 2)},
        "metodo": "simulacion_poisson",
        "prob_home_win": round(sim.home_win, 3),
        "prob_draw": round(sim.draw, 3),
        "prob_away_win": round(sim.away_win, 3),
        "prob_over_2_5": round(sim.over_2_5, 3),
        "prob_btts": round(sim.btts, 3),
        "marcador_mas_probable": f"{sim.top_scoreline[0]}-{sim.top_scoreline[1]}",
    }
    # Mercados de goles exactos (over/under, distribución, marcadores, portería a cero).
    result["mercados_goles"] = goal_markets(lh, la)

    if _MODEL_PATH.exists():
        try:  # si el modelo no carga (versión incompatible), seguimos con la simulación
            import joblib
            import numpy as np

            bundle = joblib.load(_MODEL_PATH)
            model, feats = bundle["model"], bundle["features"]
            fpd = (hp - ap) if (hp is not None and ap is not None) else None
            values = {
                "home_elo": home_elo, "away_elo": away_elo, "elo_diff": home_elo - away_elo,
                "rest_days_home": 5, "rest_days_away": 5,
                "home_form_pts": hp, "away_form_pts": ap, "form_pts_diff": fpd,
                "home_gf": hgf, "home_ga": hga, "away_gf": agf, "away_ga": aga,
                "h2h_home_ppg": h2h,
                "is_neutral": 1 if neutral else 0,
                "importance": 2 if neutral else 1,
                "home_sos": home_sos, "away_sos": away_sos,
            }
            X = np.array([[values.get(f) if values.get(f) is not None else np.nan
                           for f in feats]], dtype=float)
            p = model.predict_proba(X)[0]
            result.update(
                metodo="xgboost_1x2 + simulacion_poisson",
                prob_home_win=round(float(p[0]), 3),
                prob_draw=round(float(p[1]), 3),
                prob_away_win=round(float(p[2]), 3),
            )
        except Exception:
            pass

    # --- TARJETAS (si el modelo está entrenado y hay datos) ---
    if _CARDS_MODEL.exists() and home_cards is not None and away_cards is not None:
        try:
            from statsmodels.iolib.smpickle import load_pickle

            model = load_pickle(str(_CARDS_MODEL))
            ref_val = ref_cards if ref_cards is not None else (home_cards + away_cards) / 2
            lam = _glm_lambda(model, [ref_val, home_cards, away_cards])
            cm = count_markets(lam, [2.5, 3.5, 4.5, 5.5, 6.5], max_count=10)
            result["tarjetas"] = {
                "esperadas": cm["esperado"],
                "arbitro_usado": referee or "(media, sin árbitro)",
                "mas_probable": cm["mas_probable"],
                "over_under": cm["over_under"],
                "distribucion": cm["distribucion"],
            }
        except Exception:
            pass

    # --- CÓRNERS (si el modelo está entrenado y hay datos) ---
    if _CORNERS_MODEL.exists() and None not in (home_cf, home_ca, away_cf, away_ca):
        try:
            from statsmodels.iolib.smpickle import load_pickle

            model = load_pickle(str(_CORNERS_MODEL))
            hs = home_shots if home_shots is not None else 10.0
            as_ = away_shots if away_shots is not None else 10.0
            lam = _glm_lambda(model, [home_cf, home_ca, away_cf, away_ca, hs, as_,
                                      home_elo - away_elo])
            cm = count_markets(lam, [7.5, 8.5, 9.5, 10.5, 11.5], max_count=16)
            result["corners"] = {
                "esperados": cm["esperado"],
                "mas_probable": cm["mas_probable"],
                "over_under": cm["over_under"],
                "distribucion": cm["distribucion"],
            }
        except Exception:
            pass

    # --- TIROS A PUERTA por jugador (principales rematadores de cada equipo) ---
    if _PLAYER_SHOTS_MODEL.exists():
        try:
            from statsmodels.iolib.smpickle import load_pickle

            pmodel = load_pickle(str(_PLAYER_SHOTS_MODEL))
            with connect() as con:
                local = _predict_team_shots(con, pmodel, home_team_id, True)
                visita = _predict_team_shots(con, pmodel, away_team_id, False)
            if local or visita:
                result["tiros_jugadores"] = {"local": local, "visitante": visita}
        except Exception:
            pass

    # Comparativa de equipos (el contexto detrás de la predicción).
    def _r(x):
        return round(x, 2) if x is not None else None

    result["comparativa"] = {
        "local": {"elo": round(home_elo, 0), "forma_pts": _r(hp), "goles_favor": _r(hgf),
                  "goles_contra": _r(hga), "sos_elo_rival": _r(home_sos)},
        "visitante": {"elo": round(away_elo, 0), "forma_pts": _r(ap), "goles_favor": _r(agf),
                      "goles_contra": _r(aga), "sos_elo_rival": _r(away_sos)},
        "h2h_ppg_local": _r(h2h),
    }

    # Doble oportunidad (derivada del 1X2 final): cubre dos de los tres resultados.
    ph, pd, pa = result["prob_home_win"], result["prob_draw"], result["prob_away_win"]
    result["doble_oportunidad"] = {
        "local_o_empate_1X": round(ph + pd, 3),
        "local_o_visita_12": round(ph + pa, 3),
        "empate_o_visita_X2": round(pd + pa, 3),
    }
    return result
