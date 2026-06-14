"""Capa BRONZE: ingesta de datos CRUDOS desde api-football, sin transformar.

Regla de oro del medallón: el crudo es inmutable y nunca se borra. Si tu lógica de
limpieza tiene un bug, re-procesas Silver/Gold desde aquí SIN volver a gastar llamadas
(ni cuota) del API.

Diseño de esta capa:
  - Cada respuesta se guarda como JSON Lines (JSONL) en `data/bronze/<endpoint>/`.
  - REANUDABLE: si el archivo ya existe, se omite la llamada (salvo `force=True`). Así un
    backfill que se corta se retoma sin repetir llamadas. Incluso las respuestas vacías se
    escriben (archivo vacío) para no volver a preguntar por ellas.
  - Las funciones de bajo nivel reciben un `client` ya abierto para reutilizar la conexión
    y el rate limiter en lotes grandes.
"""

from __future__ import annotations

import json
from typing import Any

from rich import print

from worldcup.common.config import settings
from worldcup.dataplatform.api_client import ApiFootballClient


def _raw_path(endpoint: str, key: str):
    safe_endpoint = endpoint.replace("/", "_")
    folder = settings.bronze_dir / safe_endpoint
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{safe_endpoint}__{key}.jsonl"


def _is_ingested(endpoint: str, key: str) -> bool:
    """True si ya existe el archivo (aunque esté vacío) -> lote ya procesado."""
    return _raw_path(endpoint, key).exists()


def save_raw(
    endpoint: str, key: str, rows: list[dict[str, Any]], inject: dict[str, Any] | None = None
) -> int:
    """Persiste una lista de objetos crudos como JSON Lines. Devuelve nº de filas.

    `inject` añade claves a CADA fila (p.ej. {"fixture_id": 123}). Necesario porque las
    respuestas de detalle no traen el fixture_id en el cuerpo, solo en el parámetro.
    """
    out_file = _raw_path(endpoint, key)
    with out_file.open("w", encoding="utf-8") as f:
        for row in rows:
            if inject:
                row = {**inject, **row}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _ingest(
    client: ApiFootballClient,
    endpoint: str,
    key: str,
    params: dict[str, Any],
    *,
    force: bool = False,
    inject: dict[str, Any] | None = None,
) -> int:
    """Núcleo genérico: llama un endpoint y guarda el crudo. Reanudable."""
    if not force and _is_ingested(endpoint, key):
        return -1  # señal de "omitido (ya estaba)"
    rows = client.get(endpoint, params)
    return save_raw(endpoint, key, rows, inject=inject)


# ───────────────────────── Datos estáticos (una vez) ─────────────────────────


def ingest_leagues(client: ApiFootballClient, *, force: bool = False) -> int:
    """Catálogo de ligas y competiciones (ids para todo lo demás)."""
    return _ingest(client, "leagues", "all", {}, force=force)


def ingest_teams(client: ApiFootballClient, league: int, season: int, *, force: bool = False) -> int:
    """Equipos (con su `venue`) de una liga/temporada."""
    return _ingest(client, "teams", f"league{league}_season{season}", {"league": league, "season": season}, force=force)


def ingest_standings(client: ApiFootballClient, league: int, season: int, *, force: bool = False) -> int:
    """Clasificación final/parcial de la liga (feature de fuerza de equipo)."""
    return _ingest(client, "standings", f"league{league}_season{season}", {"league": league, "season": season}, force=force)


# ───────────────────────── Partidos (base) ─────────────────────────


def ingest_fixtures(league: int, season: int, *, force: bool = False) -> int:
    """Todos los partidos de una liga/temporada. Endpoint base del que cuelga todo."""
    with ApiFootballClient() as client:
        return _ingest(client, "fixtures", f"league{league}_season{season}", {"league": league, "season": season}, force=force)


# ───────────────────────── Detalle por partido (caro) ─────────────────────────
# Cada uno cuesta 1 llamada por fixture. Por eso son reanudables y se hacen en lote.


def ingest_fixture_statistics(client: ApiFootballClient, fixture_id: int, *, force: bool = False) -> int:
    """Estadísticas del partido: tiros, tiros a puerta, posesión, córners, faltas, tarjetas."""
    return _ingest(client, "fixtures/statistics", f"fixture{fixture_id}", {"fixture": fixture_id}, force=force, inject={"fixture_id": fixture_id})


def ingest_fixture_events(client: ApiFootballClient, fixture_id: int, *, force: bool = False) -> int:
    """Eventos minuto a minuto: goles (con asistente), tarjetas (con jugador), cambios."""
    return _ingest(client, "fixtures/events", f"fixture{fixture_id}", {"fixture": fixture_id}, force=force, inject={"fixture_id": fixture_id})


def ingest_fixture_lineups(client: ApiFootballClient, fixture_id: int, *, force: bool = False) -> int:
    """Alineaciones, formación y titulares/suplentes."""
    return _ingest(client, "fixtures/lineups", f"fixture{fixture_id}", {"fixture": fixture_id}, force=force, inject={"fixture_id": fixture_id})


def ingest_fixture_players(client: ApiFootballClient, fixture_id: int, *, force: bool = False) -> int:
    """Stats POR JUGADOR del partido: tiros a puerta, pases, regates, rating... (clave para
    los modelos a nivel jugador)."""
    return _ingest(client, "fixtures/players", f"fixture{fixture_id}", {"fixture": fixture_id}, force=force, inject={"fixture_id": fixture_id})


def ingest_odds(client: ApiFootballClient, fixture_id: int, *, force: bool = False) -> int:
    """Cuotas de las casas para el partido. Uno de los features más predictivos que existen."""
    return _ingest(client, "odds", f"fixture{fixture_id}", {"fixture": fixture_id}, force=force, inject={"fixture_id": fixture_id})


# Mapa de endpoints de detalle por partido -> función. Permite elegir qué ingerir.
FIXTURE_DETAIL_INGESTORS = {
    "statistics": ingest_fixture_statistics,
    "events": ingest_fixture_events,
    "lineups": ingest_fixture_lineups,
    "players": ingest_fixture_players,
    "odds": ingest_odds,
}


# ───────────────────────── Utilidades de lote ─────────────────────────


def list_fixture_ids(league: int | None = None, season: int | None = None) -> list[int]:
    """Lee los fixture_id ya ingeridos en Bronze/fixtures (opcionalmente filtrando)."""
    import polars as pl

    folder = settings.bronze_dir / "fixtures"
    if league is not None and season is not None:
        files = [folder / f"fixtures__league{league}_season{season}.jsonl"]
    else:
        files = sorted(folder.glob("*.jsonl"))
    ids: set[int] = set()
    for f in files:
        if not f.exists() or f.stat().st_size == 0:
            continue
        # infer_schema_length=None: escanea TODAS las filas para inferir tipos y evitar
        # el panic de Polars cuando un campo es nulo en las primeras filas y numérico luego.
        df = pl.read_ndjson(f, infer_schema_length=None)
        ids.update(
            df.select(pl.col("fixture").struct.field("id")).to_series().to_list()
        )
    return sorted(ids)


def backfill_fixture_details(
    league: int,
    season: int,
    include: list[str] | None = None,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Recorre TODOS los partidos de una liga/temporada y baja su detalle.

    REANUDABLE: omite lo ya ingerido. `include` selecciona qué endpoints
    (por defecto todos: statistics, events, lineups, players, odds).
    Devuelve un conteo {endpoint: nuevas_llamadas}.
    """
    include = include or list(FIXTURE_DETAIL_INGESTORS)
    fixture_ids = list_fixture_ids(league, season)
    if not fixture_ids:
        raise FileNotFoundError(
            f"No hay fixtures de league{league}/season{season} en Bronze. "
            f"Ejecuta primero: wc ingest-fixtures --league {league} --season {season}"
        )

    counts = dict.fromkeys(include, 0)
    total = len(fixture_ids)
    with ApiFootballClient() as client:
        for i, fid in enumerate(fixture_ids, 1):
            for name in include:
                n = FIXTURE_DETAIL_INGESTORS[name](client, fid, force=force)
                if n >= 0:
                    counts[name] += 1
            if i % 25 == 0 or i == total:
                print(f"  [dim]…{i}/{total} partidos[/dim]")
    return counts


def backfill_season(
    league: int,
    season: int,
    include: list[str] | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Backfill COMPLETO de una liga/temporada: estáticos + fixtures + todo el detalle."""
    summary: dict[str, Any] = {}
    with ApiFootballClient() as client:
        summary["teams"] = ingest_teams(client, league, season, force=force)
        summary["standings"] = ingest_standings(client, league, season, force=force)
        summary["fixtures"] = _ingest(
            client, "fixtures", f"league{league}_season{season}",
            {"league": league, "season": season}, force=force,
        )
    summary["detail"] = backfill_fixture_details(league, season, include, force=force)
    return summary


# TODO (fuente externa): el ranking FIFA oficial no lo expone api-football de forma
# limpia. Se complementa con otra fuente (web oficial FIFA / dataset público) y se une
# en Silver por (selección, fecha). Pendiente para selecciones.
