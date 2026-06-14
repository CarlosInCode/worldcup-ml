"""Capa SILVER: el crudo de Bronze -> tablas limpias, tipadas y deduplicadas.

Tablas que produce (todas en warehouse.duckdb):
  - matches              1 fila por partido (con etiqueta 1X2)
  - match_statistics     1 fila por (partido, equipo): tiros, córners, faltas, tarjetas...
  - match_events         1 fila por evento: goles, tarjetas (con jugador y minuto), cambios
  - player_match_stats   1 fila por (partido, equipo, jugador): tiros a puerta, rating...

Aquí NO se calculan features de ML (eso es Gold); aquí solo se normaliza y se tipa.
Cada builder es idempotente (CREATE OR REPLACE) y tolera que falten datos en Bronze.
"""

from __future__ import annotations

import re

import polars as pl

from worldcup.common.config import settings
from worldcup.common.db import connect


def _read_jsonl_dir(subdir: str) -> list[dict]:
    """Lee todas las filas de todos los JSONL de data/bronze/<subdir>/."""
    folder = settings.bronze_dir / subdir
    rows: list[dict] = []
    if not folder.exists():
        return rows
    import json

    for f in sorted(folder.glob("*.jsonl")):
        if f.stat().st_size == 0:
            continue
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _snake(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _num(value):
    """Convierte valores del API a número: '82%' -> 82, None -> None, '12' -> 12."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip().rstrip("%")
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return None


def _write(table: str, df: pl.DataFrame) -> int:
    with connect() as con:
        con.register("_df", df.to_arrow())
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _df")
        return int(con.sql(f"SELECT count(*) FROM {table}").fetchone()[0])


# ───────────────────────── matches ─────────────────────────


def _classify_competition(name: str) -> tuple[str, bool]:
    """Clasifica una competición -> (tipo, es_sede_neutral).

    tipo ∈ {friendly, qualifier, nations, tournament, league}.
    Sede neutral: torneos a sede única (Mundial, Eurocopa, Copa América...), NO sus
    clasificatorias ni la Nations League (esas son ida/vuelta en casa de cada uno).
    """
    n = name.lower()
    if "qualif" in n:
        return "qualifier", False
    if "nations league" in n:
        return "nations", False
    if "friendl" in n:
        return "friendly", False
    tournament_kw = ["world cup", "euro championship", "copa america", "copa américa",
                     "africa cup", "asian cup", "gold cup", "confederations"]
    if any(k in n for k in tournament_kw):
        return "tournament", True  # fase final a sede única -> neutral
    return "league", False


def build_league_meta() -> int:
    """Tabla `league_meta`: clasifica cada liga (tipo, neutral) para usarla como feature."""
    from worldcup.dataplatform import catalog

    try:
        cat = catalog.load_catalog()
    except FileNotFoundError:
        return 0
    rows = []
    for lg in cat:
        ctype, neutral = _classify_competition(lg.name)
        rows.append({
            "league_id": lg.league_id, "league_name": lg.name,
            "competition_type": ctype, "is_neutral": neutral,
        })
    return _write("league_meta", pl.DataFrame(rows).unique(subset=["league_id"]))


def build_matches() -> int:
    """Tabla `matches`: 1 fila por partido, con la etiqueta 1X2."""
    files = sorted((settings.bronze_dir / "fixtures").glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(
            "No hay datos en data/bronze/fixtures. Ejecuta antes la ingesta "
            "(wc seed --no-detail)."
        )
    frames = [pl.read_ndjson(f, infer_schema_length=None) for f in files if f.stat().st_size]
    raw = pl.concat(frames, how="diagonal_relaxed")

    matches = raw.select(
        pl.col("fixture").struct.field("id").alias("match_id"),
        pl.col("fixture").struct.field("date").alias("kickoff_utc"),
        pl.col("league").struct.field("id").alias("league_id"),
        pl.col("league").struct.field("season").alias("season"),
        pl.col("teams").struct.field("home").struct.field("id").alias("home_team_id"),
        pl.col("teams").struct.field("home").struct.field("name").alias("home_team"),
        pl.col("teams").struct.field("away").struct.field("id").alias("away_team_id"),
        pl.col("teams").struct.field("away").struct.field("name").alias("away_team"),
        pl.col("goals").struct.field("home").alias("home_goals"),
        pl.col("goals").struct.field("away").alias("away_goals"),
        pl.col("fixture").struct.field("status").struct.field("short").alias("status"),
        pl.col("fixture").struct.field("referee").alias("referee"),
    ).unique(subset=["match_id"])

    matches = matches.filter(pl.col("status") == "FT")
    matches = matches.with_columns(
        pl.when(pl.col("home_goals") > pl.col("away_goals"))
        .then(0)
        .when(pl.col("home_goals") == pl.col("away_goals"))
        .then(1)
        .otherwise(2)
        .alias("result_1x2")
    ).with_columns(
        # api-football entrega fechas ISO con zona horaria (+00:00) -> formato explícito.
        pl.col("kickoff_utc").str.to_datetime(format="%Y-%m-%dT%H:%M:%S%z", strict=False)
    )
    return _write("matches", matches)


# ───────────────────────── match_statistics ─────────────────────────


def build_match_statistics() -> int:
    """Tabla `match_statistics`: 1 fila por (partido, equipo). Pivotea los tipos de stat
    a columnas (shots_on_goal, total_shots, corner_kicks, fouls, yellow_cards...)."""
    raw = _read_jsonl_dir("fixtures_statistics")
    if not raw:
        return 0
    rows = []
    for item in raw:
        team = item.get("team") or {}
        rec = {
            "match_id": item.get("fixture_id"),
            "team_id": team.get("id"),
            "team_name": team.get("name"),
        }
        for stat in item.get("statistics") or []:
            rec[_snake(stat.get("type", ""))] = _num(stat.get("value"))
        rows.append(rec)
    df = pl.DataFrame(rows, infer_schema_length=None).unique(subset=["match_id", "team_id"])
    return _write("match_statistics", df)


# ───────────────────────── match_events ─────────────────────────


def build_match_events() -> int:
    """Tabla `match_events`: 1 fila por evento (goles, tarjetas con jugador/minuto, cambios)."""
    raw = _read_jsonl_dir("fixtures_events")
    if not raw:
        return 0
    rows = []
    for e in raw:
        time = e.get("time") or {}
        team = e.get("team") or {}
        player = e.get("player") or {}
        assist = e.get("assist") or {}
        rows.append(
            {
                "match_id": e.get("fixture_id"),
                "minute": time.get("elapsed"),
                "minute_extra": time.get("extra"),
                "team_id": team.get("id"),
                "team_name": team.get("name"),
                "player_id": player.get("id"),
                "player_name": player.get("name"),
                "assist_player_id": assist.get("id"),
                "type": e.get("type"),       # Goal | Card | subst | Var
                "detail": e.get("detail"),   # 'Yellow Card' | 'Red Card' | 'Normal Goal'...
            }
        )
    df = pl.DataFrame(rows, infer_schema_length=None)
    return _write("match_events", df)


# ───────────────────────── player_match_stats ─────────────────────────


def build_player_match_stats() -> int:
    """Tabla `player_match_stats`: 1 fila por (partido, equipo, jugador) con sus stats.

    Clave para los modelos a nivel jugador: tiros totales / a puerta, goles, minutos, rating.
    """
    raw = _read_jsonl_dir("fixtures_players")
    if not raw:
        return 0
    rows = []
    for block in raw:
        match_id = block.get("fixture_id")
        team = block.get("team") or {}
        for p in block.get("players") or []:
            player = p.get("player") or {}
            stats = (p.get("statistics") or [{}])[0]
            games = stats.get("games") or {}
            shots = stats.get("shots") or {}
            goals = stats.get("goals") or {}
            passes = stats.get("passes") or {}
            cards = stats.get("cards") or {}
            rows.append(
                {
                    "match_id": match_id,
                    "team_id": team.get("id"),
                    "player_id": player.get("id"),
                    "player_name": player.get("name"),
                    "position": games.get("position"),
                    "minutes": _num(games.get("minutes")),
                    "rating": _num(games.get("rating")),
                    "shots_total": _num(shots.get("total")),
                    "shots_on": _num(shots.get("on")),
                    "goals_total": _num(goals.get("total")),
                    "goals_assists": _num(goals.get("assists")),
                    "passes_total": _num(passes.get("total")),
                    "passes_accuracy": _num(passes.get("accuracy")),
                    "cards_yellow": _num(cards.get("yellow")),
                    "cards_red": _num(cards.get("red")),
                }
            )
    if not rows:
        return 0
    df = pl.DataFrame(rows, infer_schema_length=None).unique(
        subset=["match_id", "player_id"]
    )
    return _write("player_match_stats", df)


# ───────────────────────── orquestador ─────────────────────────


def build_all() -> dict[str, int]:
    """Construye TODAS las tablas Silver que tengan datos en Bronze."""
    counts = {"matches": build_matches()}  # matches es obligatorio
    for name, fn in [
        ("league_meta", build_league_meta),
        ("match_statistics", build_match_statistics),
        ("match_events", build_match_events),
        ("player_match_stats", build_player_match_stats),
    ]:
        counts[name] = fn()
    return counts
