"""Orquestador de ingesta: decide QUÉ ligas/temporadas traer y las baja con presupuesto.

Pensado para predecir el Mundial 2026 y para tu cuota diaria (p.ej. 7.500 llamadas/día):

  - Prioriza SELECCIONES (Mundial, eliminatorias, Nations League, Eurocopa, Copa América,
    AFCON, Asian Cup, Gold Cup): son los datos más relevantes y relativamente pocos.
  - Las LIGAS DE CLUBES son opcionales (scope="clubs"/"all"): caras, sirven sobre todo para
    modelos a nivel jugador.
  - DOS PASADAS: (1) fixtures = todos los resultados, baratísimo (~1 llamada por
    liga-temporada); (2) detalle por partido = caro, con TOPE DE LLAMADAS.
  - REANUDABLE + PRESUPUESTADO: si llegas al tope diario se detiene limpio; mañana lo vuelves
    a lanzar y retoma exactamente donde quedó sin repetir llamadas.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich import print

from worldcup.dataplatform import bronze, catalog
from worldcup.dataplatform.api_client import ApiFootballClient


@dataclass
class _Spec:
    name: str
    country: str | None = None
    match: str = "exact"
    detail: bool = True  # ¿bajar el detalle caro por partido de esta competición?


# Competiciones de SELECCIONES — el corazón para predecir el Mundial.
NATIONAL: list[_Spec] = [
    _Spec("World Cup", country="World"),
    _Spec("World Cup - Qualification", match="contains"),
    _Spec("UEFA Nations League"),
    _Spec("Euro Championship", match="contains"),  # incluye su clasificación
    _Spec("Copa America"),
    _Spec("Africa Cup of Nations", match="contains"),
    _Spec("Asian Cup", match="contains"),
    _Spec("Gold Cup", match="contains"),
    _Spec("CONCACAF Nations League", match="contains"),
    # Amistosos: resultados SÍ (útiles para Elo), pero detalle NO (cobertura pobre + volumen enorme).
    _Spec("Friendlies", country="World", detail=False),
]

# Ligas de CLUBES — opcionales. Caras pero clave para features de jugador.
CLUB: list[_Spec] = [
    _Spec("Premier League", country="England"),
    _Spec("La Liga", country="Spain"),
    _Spec("Serie A", country="Italy"),
    _Spec("Bundesliga", country="Germany"),
    _Spec("Ligue 1", country="France"),
    _Spec("UEFA Champions League"),
    _Spec("Eredivisie", country="Netherlands"),
    _Spec("Primeira Liga", country="Portugal"),
    _Spec("Serie A", country="Brazil"),
    _Spec("Major League Soccer", country="USA"),
]


@dataclass
class _Target:
    league_id: int
    season: int
    name: str
    detail: bool


def _build_targets(scope: str, season_min: int, season_max: int) -> list[_Target]:
    cat = catalog.load_catalog()
    specs = {"national": NATIONAL, "clubs": CLUB, "all": NATIONAL + CLUB}[scope]
    wanted = set(range(season_min, season_max + 1))

    targets: list[_Target] = []
    unresolved: list[str] = []
    for spec in specs:
        matches = catalog.resolve(cat, spec.name, country=spec.country, match=spec.match)
        if not matches:
            unresolved.append(spec.name)
            continue
        for lg in matches:
            # Solo temporadas que el catálogo dice que existen -> cero llamadas malgastadas.
            for season in sorted(wanted & set(lg.seasons)):
                targets.append(_Target(lg.league_id, season, lg.name, spec.detail))

    if unresolved:
        print(f"[yellow]⚠ No se resolvieron (revisa el catálogo): {unresolved}[/yellow]")
    return targets


def run_seed(
    scope: str = "national",
    seasons: tuple[int, int] = (2018, 2026),
    endpoints: list[str] | None = None,
    max_calls: int = 7000,
    do_detail: bool = True,
) -> dict:
    """Ejecuta el plan de ingesta completo respetando el presupuesto de llamadas.

    scope: "national" (defecto) | "clubs" | "all".
    seasons: rango (min, max) inclusive.
    endpoints: detalle por partido a bajar (defecto: statistics + events).
    max_calls: tope de llamadas para ESTA ejecución (déjalo por debajo de tu cuota diaria).
    do_detail: si False, solo hace la pasada barata de fixtures (todos los resultados).
    """
    endpoints = endpoints or ["statistics", "events"]
    season_min, season_max = seasons
    targets = _build_targets(scope, season_min, season_max)
    if not targets:
        raise RuntimeError("No se resolvió ninguna liga/temporada. ¿Ejecutaste 'wc ingest-leagues'?")

    print(f"[bold]Plan:[/bold] scope={scope}, temporadas {season_min}–{season_max}, "
          f"{len(targets)} liga-temporadas, presupuesto={max_calls} llamadas.")

    summary: dict = {"fixtures_seasons": 0, "fixtures_total": 0, "detail_calls": 0, "stopped_at_budget": False}

    with ApiFootballClient() as client:
        # ── PASADA 1: fixtures + equipos + clasificación (barato) ──
        print("[bold cyan]Pasada 1[/bold cyan] — fixtures (resultados de todos los partidos)")
        for t in targets:
            if client.calls_made >= max_calls:
                summary["stopped_at_budget"] = True
                break
            bronze.ingest_teams(client, t.league_id, t.season)
            bronze.ingest_standings(client, t.league_id, t.season)
            bronze._ingest(
                client, "fixtures", f"league{t.league_id}_season{t.season}",
                {"league": t.league_id, "season": t.season},
            )
            summary["fixtures_seasons"] += 1

        # Cuántos partidos tenemos ya (sirve para estimar el costo del detalle).
        detail_targets = [t for t in targets if t.detail]
        all_fids: list[tuple[int, int]] = []  # (fixture_id) con detalle pendiente
        for t in detail_targets:
            fids = bronze.list_fixture_ids(t.league_id, t.season)
            all_fids.extend(fids)
            summary["fixtures_total"] += len(fids)

        est = len(all_fids) * len(endpoints)
        print(f"[dim]Llamadas usadas hasta ahora: {client.calls_made}. "
              f"Partidos con detalle pendiente: {len(all_fids)}. "
              f"Estimado detalle completo: ~{est} llamadas ({endpoints}).[/dim]")

        if not do_detail:
            print("[green]✓[/green] Solo fixtures (--no-detail). Ya tienes todos los resultados.")
            summary["calls_made"] = client.calls_made
            return summary

        # ── PASADA 2: detalle por partido (caro, con tope de presupuesto) ──
        print("[bold magenta]Pasada 2[/bold magenta] — detalle por partido (con presupuesto)")
        done = 0
        summary["errors"] = 0
        for fid in all_fids:
            if client.calls_made >= max_calls:
                summary["stopped_at_budget"] = True
                break
            for name in endpoints:
                if client.calls_made >= max_calls:
                    summary["stopped_at_budget"] = True
                    break
                try:
                    n = bronze.FIXTURE_DETAIL_INGESTORS[name](client, fid)
                    if n >= 0:
                        summary["detail_calls"] += 1
                except Exception as e:
                    # RESILIENCIA: un partido/endpoint problemático no debe tumbar todo el
                    # backfill. Lo saltamos (se reintentará en la próxima corrida) y seguimos.
                    summary["errors"] += 1
                    if summary["errors"] <= 10:
                        print(f"  [yellow]⚠ {name} fixture {fid}: {str(e)[:80]}[/yellow]")
            done += 1
            if done % 50 == 0:
                print(f"  [dim]…{done}/{len(all_fids)} partidos, {client.calls_made} llamadas, "
                      f"{summary['errors']} errores[/dim]")

    summary["calls_made"] = client.calls_made
    if summary["stopped_at_budget"]:
        print(f"[yellow]⏸ Tope de {max_calls} llamadas alcanzado. Vuelve a ejecutar mañana: "
              f"retoma donde quedó sin repetir.[/yellow]")
    else:
        print("[green]✓[/green] Ingesta completada dentro del presupuesto.")
    return summary
