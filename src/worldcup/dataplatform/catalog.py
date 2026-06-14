"""Resolución del catálogo de ligas de api-football.

En vez de hardcodear ids de liga (frágiles y fáciles de equivocar), los resolvemos POR
NOMBRE desde el catálogo `leagues` que ya ingerimos en Bronze. Además, el catálogo nos
dice qué temporadas existen para cada competición, así evitamos gastar llamadas en
temporadas que no se jugaron.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from worldcup.common.config import settings


@dataclass
class ResolvedLeague:
    league_id: int
    name: str
    country: str
    type: str
    seasons: list[int]


def _leagues_file():
    return settings.bronze_dir / "leagues" / "leagues__all.jsonl"


def load_catalog() -> list[ResolvedLeague]:
    """Lee el catálogo crudo de Bronze y lo devuelve como lista de ResolvedLeague."""
    f = _leagues_file()
    if not f.exists() or f.stat().st_size == 0:
        raise FileNotFoundError(
            "Falta el catálogo de ligas. Ejecuta primero: wc ingest-leagues"
        )
    raw = pl.read_ndjson(f)
    out: list[ResolvedLeague] = []
    for row in raw.iter_rows(named=True):
        lg = row["league"]
        country = row.get("country") or {}
        seasons = [int(s["year"]) for s in (row.get("seasons") or [])]
        out.append(
            ResolvedLeague(
                league_id=int(lg["id"]),
                name=lg["name"],
                country=(country.get("name") or "World"),
                type=lg.get("type") or "",
                seasons=sorted(seasons),
            )
        )
    return out


def resolve(
    catalog: list[ResolvedLeague],
    name: str,
    *,
    country: str | None = None,
    match: str = "exact",
) -> list[ResolvedLeague]:
    """Encuentra ligas por nombre (y país opcional).

    match="exact": nombre idéntico (case-insensitive).
    match="contains": el nombre contiene el texto (p.ej. "World Cup - Qualification"
                      devuelve las eliminatorias de TODAS las confederaciones).
    """
    name_l = name.lower()
    res = []
    for lg in catalog:
        n = lg.name.lower()
        ok_name = (n == name_l) if match == "exact" else (name_l in n)
        ok_country = country is None or lg.country.lower() == country.lower()
        if ok_name and ok_country:
            res.append(lg)
    return res
