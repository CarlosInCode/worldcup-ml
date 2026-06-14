"""Monitoreo de CALIDAD de datos: validaciones sobre las tablas Silver y Gold.

Mejor práctica clave de ingeniería de datos: nunca entrenes sobre datos que no validaste.
Estas comprobaciones se ejecutan como "puertas de calidad" entre etapas del pipeline: si
una comprobación CRÍTICA falla, el pipeline se detiene antes de propagar basura a los modelos.

Cada check devuelve un CheckResult (nombre, ok, severidad, detalle). `wc check` los corre todos.
"""

from __future__ import annotations

from dataclasses import dataclass

from worldcup.common.db import connect


@dataclass
class CheckResult:
    name: str
    ok: bool
    critical: bool
    detail: str


def _table_count(con, table: str) -> int:
    exists = con.sql(
        f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table}'"
    ).fetchone()[0]
    return int(con.sql(f"SELECT count(*) FROM {table}").fetchone()[0]) if exists else -1


def _check(con, name, sql, *, critical=True, expect_zero=True) -> CheckResult:
    """Ejecuta una consulta que cuenta filas 'malas'. ok = (conteo == 0)."""
    bad = int(con.sql(sql).fetchone()[0])
    ok = (bad == 0) if expect_zero else (bad > 0)
    return CheckResult(name, ok, critical, f"{bad} filas" if expect_zero else f"{bad}")


def run_checks() -> list[CheckResult]:
    """Corre todas las validaciones disponibles según las tablas que existan."""
    results: list[CheckResult] = []
    with connect() as con:
        n_matches = _table_count(con, "matches")

        # --- matches (Silver) ---
        if n_matches <= 0:
            results.append(CheckResult("matches: existe y no vacía", False, True, "tabla vacía/ausente"))
            return results
        results.append(CheckResult("matches: filas > 0", True, True, f"{n_matches} filas"))
        results += [
            _check(con, "matches: sin match_id nulo", "SELECT count(*) FROM matches WHERE match_id IS NULL"),
            _check(con, "matches: sin match_id duplicado",
                   "SELECT count(*) FROM (SELECT match_id FROM matches GROUP BY match_id HAVING count(*)>1)"),
            _check(con, "matches: result_1x2 en {0,1,2}",
                   "SELECT count(*) FROM matches WHERE result_1x2 NOT IN (0,1,2)"),
            _check(con, "matches: goles no negativos",
                   "SELECT count(*) FROM matches WHERE home_goals < 0 OR away_goals < 0"),
            _check(con, "matches: kickoff no nulo",
                   "SELECT count(*) FROM matches WHERE kickoff_utc IS NULL", critical=False),
        ]

        # --- match_features (Gold) ---
        if _table_count(con, "match_features") > 0:
            results += [
                _check(con, "match_features: 1 fila por match_id (sin duplicar)",
                       "SELECT count(*) FROM (SELECT match_id FROM match_features GROUP BY match_id HAVING count(*)>1)"),
                _check(con, "match_features: Elo en rango sano [800,2400]",
                       "SELECT count(*) FROM match_features WHERE home_elo < 800 OR home_elo > 2400 OR away_elo < 800 OR away_elo > 2400"),
                _check(con, "match_features: sin fuga (todo match en matches)",
                       "SELECT count(*) FROM match_features f LEFT JOIN matches m USING(match_id) WHERE m.match_id IS NULL"),
            ]

        # --- match_statistics (Silver, detalle) ---
        if _table_count(con, "match_statistics") > 0:
            results.append(
                _check(con, "match_statistics: córners/tarjetas no negativos",
                       "SELECT count(*) FROM match_statistics WHERE corner_kicks < 0 OR yellow_cards < 0",
                       critical=False)
            )

        # --- card_features (Gold) ---
        if _table_count(con, "card_features") > 0:
            results.append(
                _check(con, "card_features: total_cards no negativo",
                       "SELECT count(*) FROM card_features WHERE total_cards < 0")
            )
    return results


def gate(results: list[CheckResult]) -> bool:
    """True si ninguna comprobación CRÍTICA falló (la puerta de calidad se abre)."""
    return all(r.ok for r in results if r.critical)
