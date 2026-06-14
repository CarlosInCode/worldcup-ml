"""Capa GOLD: tablas listas para ML. Una fila por partido con features + etiqueta.

🔴 REGLA ANTI-LEAKAGE: todo feature de un partido se calcula usando EXCLUSIVAMENTE
información de partidos ANTERIORES. Por eso recorremos en orden cronológico y registramos
el estado PRE-partido antes de actualizarlo con el resultado.

Produce dos tablas Gold:
  - match_features  -> objetivo 1X2 (resultado). Features: Elo, forma reciente, H2H, descanso.
  - card_features   -> objetivo "total de tarjetas". Features: árbitro, tarjetas recientes
                       de cada equipo, competitividad. (Requiere match_statistics; si está
                       vacía, devuelve 0 filas hasta que bajes el detalle.)
"""

from __future__ import annotations

from collections import deque

import polars as pl

from worldcup.common import constants as K
from worldcup.common.db import connect

_BASE_ELO = K.ELO_BASE
_K = K.ELO_K
_HOME_ADVANTAGE = K.HOME_ADVANTAGE_ELO
_FORM_N = K.FORM_WINDOW


def _expected(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _count(con, table: str) -> int:
    """Nº de filas de una tabla, o 0 si la tabla aún no existe (detalle no descargado)."""
    exists = con.sql(
        f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table}'"
    ).fetchone()[0]
    return int(con.sql(f"SELECT count(*) FROM {table}").fetchone()[0]) if exists else 0


def _avg(d: deque) -> float | None:
    return sum(d) / len(d) if d else None


def build_match_features() -> int:
    """Tabla Gold `match_features` (objetivo 1X2) con features enriquecidos."""
    with connect() as con:
        has_meta = _count(con, "league_meta") > 0
        join = "LEFT JOIN league_meta lm USING(league_id)" if has_meta else ""
        cols = "m.*, lm.is_neutral, lm.competition_type" if has_meta else "m.*"
        matches = con.sql(
            f"SELECT {cols} FROM matches m {join} "
            "WHERE m.home_goals IS NOT NULL ORDER BY m.kickoff_utc"
        ).pl()

    elo: dict[int, float] = {}
    last_played: dict[int, object] = {}
    # Forma reciente por equipo: puntos, goles a favor, goles en contra (últimos N).
    pts_hist: dict[int, deque] = {}
    gf_hist: dict[int, deque] = {}
    ga_hist: dict[int, deque] = {}
    sos_hist: dict[int, deque] = {}  # strength-of-schedule: Elo de rivales enfrentados
    # Head-to-head: por par de equipos, lista de (equipo_local_de_ese_partido, result_1x2).
    h2h: dict[tuple, list] = {}
    rows = []

    def _form(team_id: int):
        return (
            _avg(pts_hist.get(team_id, deque())),
            _avg(gf_hist.get(team_id, deque())),
            _avg(ga_hist.get(team_id, deque())),
        )

    for m in matches.iter_rows(named=True):
        h, a = m["home_team_id"], m["away_team_id"]
        eh, ea = elo.get(h, _BASE_ELO), elo.get(a, _BASE_ELO)
        ko = m["kickoff_utc"]
        r = m["result_1x2"]
        neutral = bool(m.get("is_neutral")) if has_meta else False
        ctype = m.get("competition_type") if has_meta else None
        importance = K.IMPORTANCE.get(ctype, 1)
        home_adv = 0.0 if neutral else _HOME_ADVANTAGE  # sin ventaja de local en sede neutral

        rest_h = (ko - last_played[h]).days if h in last_played else None
        rest_a = (ko - last_played[a]).days if a in last_played else None
        hp, hgf, hga = _form(h)
        ap, agf, aga = _form(a)

        # Head-to-head: puntos por partido del LOCAL actual en enfrentamientos pasados.
        pair = (min(h, a), max(h, a))
        past = h2h.get(pair, [])
        if past:
            pts = 0
            for hist_home, hist_r in past:
                if hist_r == 1:
                    pts += 1
                elif hist_r == 0:
                    pts += 3 if hist_home == h else 0
                else:
                    pts += 0 if hist_home == h else 3
            h2h_home_ppg = pts / len(past)
        else:
            h2h_home_ppg = None

        rows.append(
            {
                "match_id": m["match_id"],
                "kickoff_utc": ko,
                "league_id": m["league_id"],
                "season": m["season"],
                "home_team_id": h,
                "away_team_id": a,
                "home_elo": eh,
                "away_elo": ea,
                "elo_diff": eh - ea,
                "rest_days_home": rest_h,
                "rest_days_away": rest_a,
                "home_form_pts": hp,
                "away_form_pts": ap,
                "home_gf": hgf,
                "home_ga": hga,
                "away_gf": agf,
                "away_ga": aga,
                "form_pts_diff": (hp - ap) if (hp is not None and ap is not None) else None,
                "h2h_home_ppg": h2h_home_ppg,
                "is_neutral": int(neutral),
                "importance": importance,
                "home_sos": _avg(sos_hist.get(h, deque())),  # Elo medio de rivales recientes
                "away_sos": _avg(sos_hist.get(a, deque())),
                "result_1x2": r,
            }
        )

        # --- Actualización POST-partido (home_adv = 0 si sede neutral) ---
        exp_h = _expected(eh + home_adv, ea)
        score_h = 1.0 if r == 0 else 0.5 if r == 1 else 0.0
        elo[h] = eh + _K * (score_h - exp_h)
        elo[a] = ea + _K * ((1.0 - score_h) - (1.0 - exp_h))
        last_played[h] = last_played[a] = ko
        # strength-of-schedule: registra el Elo (pre-partido) del rival enfrentado.
        sos_hist.setdefault(h, deque(maxlen=K.SOS_WINDOW)).append(ea)
        sos_hist.setdefault(a, deque(maxlen=K.SOS_WINDOW)).append(eh)

        hg, ag = m["home_goals"], m["away_goals"]
        for tid, gf, ga, pts in [
            (h, hg, ag, 3 if r == 0 else 1 if r == 1 else 0),
            (a, ag, hg, 3 if r == 2 else 1 if r == 1 else 0),
        ]:
            pts_hist.setdefault(tid, deque(maxlen=_FORM_N)).append(pts)
            gf_hist.setdefault(tid, deque(maxlen=_FORM_N)).append(gf)
            ga_hist.setdefault(tid, deque(maxlen=_FORM_N)).append(ga)
        h2h.setdefault(pair, []).append((h, r))

    features = pl.DataFrame(rows, infer_schema_length=None)
    with connect() as con:
        con.register("feat_df", features.to_arrow())
        con.execute("CREATE OR REPLACE TABLE match_features AS SELECT * FROM feat_df")
        return int(con.sql("SELECT count(*) FROM match_features").fetchone()[0])


def build_card_features() -> int:
    """Tabla Gold `card_features` (objetivo: total de tarjetas del partido).

    Requiere `match_statistics` con datos (baja el detalle: wc seed --scope national).
    El feature estrella es el promedio de tarjetas del ÁRBITRO en partidos anteriores.
    """
    with connect() as con:
        if not _count(con, "match_statistics"):
            return 0
        # Tarjetas por partido (suma de ambos equipos), unidas a matches (árbitro, equipos, fecha).
        df = con.sql(
            """
            SELECT m.match_id, m.kickoff_utc, m.referee,
                   m.home_team_id, m.away_team_id, s.cards
            FROM matches m
            JOIN (
                SELECT match_id,
                       sum(coalesce(yellow_cards,0)) + sum(coalesce(red_cards,0)) AS cards
                FROM match_statistics GROUP BY match_id
            ) s ON s.match_id = m.match_id
            WHERE m.home_goals IS NOT NULL
            ORDER BY m.kickoff_utc
            """
        ).pl()

    ref_hist: dict[str, deque] = {}
    team_cards: dict[int, deque] = {}
    rows = []
    for r in df.iter_rows(named=True):
        ref = r["referee"]
        h, a = r["home_team_id"], r["away_team_id"]
        rows.append(
            {
                "match_id": r["match_id"],
                "kickoff_utc": r["kickoff_utc"],
                "referee": ref,
                "ref_avg_cards": _avg(ref_hist.get(ref, deque())) if ref else None,
                "home_avg_cards": _avg(team_cards.get(h, deque())),
                "away_avg_cards": _avg(team_cards.get(a, deque())),
                "total_cards": r["cards"],  # etiqueta
            }
        )
        if ref:
            ref_hist.setdefault(ref, deque(maxlen=20)).append(r["cards"])
        # Aproximación: repartimos las tarjetas del partido a ambos equipos para su histórico.
        team_cards.setdefault(h, deque(maxlen=_FORM_N)).append(r["cards"])
        team_cards.setdefault(a, deque(maxlen=_FORM_N)).append(r["cards"])

    features = pl.DataFrame(rows, infer_schema_length=None)
    with connect() as con:
        con.register("card_df", features.to_arrow())
        con.execute("CREATE OR REPLACE TABLE card_features AS SELECT * FROM card_df")
        return int(con.sql("SELECT count(*) FROM card_features").fetchone()[0])


_CORNER_N = 10  # ventana más larga: los córners por partido son ruidosos


def build_corner_features() -> int:
    """Tabla Gold `corner_features` (objetivo: total de córners del partido).

    Los córners reflejan VOLUMEN OFENSIVO y DOMINANCIA. Features (rolling, anti-leakage):
      - córners a favor / en contra de cada equipo
      - TIROS a favor de cada equipo (volumen ofensivo -> genera córners)
      - diferencia de Elo (un equipo dominante ataca más)
    Requiere `match_statistics` (y `match_features` para el Elo).
    """
    with connect() as con:
        if not _count(con, "match_statistics"):
            return 0
        df = con.sql(
            """
            SELECT m.match_id, m.kickoff_utc, m.home_team_id, m.away_team_id,
                   hc.corner_kicks AS home_corners, ac.corner_kicks AS away_corners,
                   hc.total_shots AS home_shots, ac.total_shots AS away_shots,
                   f.elo_diff AS elo_diff
            FROM matches m
            JOIN match_statistics hc ON hc.match_id = m.match_id AND hc.team_id = m.home_team_id
            JOIN match_statistics ac ON ac.match_id = m.match_id AND ac.team_id = m.away_team_id
            LEFT JOIN match_features f ON f.match_id = m.match_id
            WHERE m.home_goals IS NOT NULL
              AND hc.corner_kicks IS NOT NULL AND ac.corner_kicks IS NOT NULL
            ORDER BY m.kickoff_utc
            """
        ).pl()

    cf: dict[int, deque] = {}   # córners a favor
    ca: dict[int, deque] = {}   # córners en contra
    sf: dict[int, deque] = {}   # tiros a favor
    rows = []

    def push(d, k, v):
        if v is not None:
            d.setdefault(k, deque(maxlen=_CORNER_N)).append(v)

    for r in df.iter_rows(named=True):
        h, a = r["home_team_id"], r["away_team_id"]
        hc, ac = r["home_corners"], r["away_corners"]
        rows.append(
            {
                "match_id": r["match_id"],
                "kickoff_utc": r["kickoff_utc"],
                "home_corners_for": _avg(cf.get(h, deque())),
                "home_corners_against": _avg(ca.get(h, deque())),
                "away_corners_for": _avg(cf.get(a, deque())),
                "away_corners_against": _avg(ca.get(a, deque())),
                "home_shots_for": _avg(sf.get(h, deque())),
                "away_shots_for": _avg(sf.get(a, deque())),
                "elo_diff": r["elo_diff"],
                "total_corners": hc + ac,  # etiqueta
            }
        )
        push(cf, h, hc)
        push(ca, h, ac)
        push(sf, h, r["home_shots"])
        push(cf, a, ac)
        push(ca, a, hc)
        push(sf, a, r["away_shots"])

    features = pl.DataFrame(rows, infer_schema_length=None)
    with connect() as con:
        con.register("corner_df", features.to_arrow())
        con.execute("CREATE OR REPLACE TABLE corner_features AS SELECT * FROM corner_df")
        return int(con.sql("SELECT count(*) FROM corner_features").fetchone()[0])


def build_player_shot_features() -> int:
    """Tabla Gold `player_shot_features` (objetivo: tiros a puerta de un jugador en el partido).

    Nivel jugador: predice los tiros a puerta de cada jugador a partir de su ritmo reciente
    (rolling) y los minutos esperados. Requiere `player_match_stats` -> necesita el endpoint
    `players` (wc seed --scope national --endpoints players).
    """
    with connect() as con:
        if not _count(con, "player_match_stats"):
            return 0
        df = con.sql(
            """
            SELECT p.match_id, m.kickoff_utc, p.player_id, p.team_id,
                   CASE WHEN p.team_id = m.home_team_id THEN 1 ELSE 0 END AS is_home,
                   p.minutes, p.shots_total, p.shots_on
            FROM player_match_stats p
            JOIN matches m ON m.match_id = p.match_id
            WHERE p.shots_on IS NOT NULL AND m.home_goals IS NOT NULL
              AND p.player_id IS NOT NULL AND p.player_id != 0  -- evita mezclar jugadores sin id
            ORDER BY m.kickoff_utc
            """
        ).pl()

    son_hist: dict[int, deque] = {}   # tiros a puerta
    sht_hist: dict[int, deque] = {}   # tiros totales
    min_hist: dict[int, deque] = {}   # minutos
    rows = []
    for r in df.iter_rows(named=True):
        pid = r["player_id"]
        rows.append(
            {
                "match_id": r["match_id"],
                "kickoff_utc": r["kickoff_utc"],
                "player_id": pid,
                "is_home": r["is_home"],
                "player_shots_on_avg": _avg(son_hist.get(pid, deque())),
                "player_shots_total_avg": _avg(sht_hist.get(pid, deque())),
                "player_minutes_avg": _avg(min_hist.get(pid, deque())),
                "shots_on": r["shots_on"],  # etiqueta
            }
        )
        son_hist.setdefault(pid, deque(maxlen=_FORM_N)).append(r["shots_on"] or 0)
        sht_hist.setdefault(pid, deque(maxlen=_FORM_N)).append(r["shots_total"] or 0)
        min_hist.setdefault(pid, deque(maxlen=_FORM_N)).append(r["minutes"] or 0)

    features = pl.DataFrame(rows, infer_schema_length=None)
    with connect() as con:
        con.register("ps_df", features.to_arrow())
        con.execute("CREATE OR REPLACE TABLE player_shot_features AS SELECT * FROM ps_df")
        return int(con.sql("SELECT count(*) FROM player_shot_features").fetchone()[0])
