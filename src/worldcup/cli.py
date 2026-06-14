"""CLI del proyecto. Ejecuta el pipeline completo paso a paso.

Uso (tras `uv sync`):
    uv run wc --help
    uv run wc ingest-fixtures --league 39 --season 2023
    uv run wc build-silver
    uv run wc build-features
    uv run wc train
    uv run wc predict --home 50 --away 33
"""

from __future__ import annotations

import json

import typer
from rich import print

app = typer.Typer(help="Plataforma de datos + ML para el Mundial FIFA 2026.")


@app.command()
def ingest_fixtures(league: int = typer.Option(...), season: int = typer.Option(...)):
    """[BRONZE] Ingesta los partidos de una liga/temporada (crudo)."""
    from worldcup.dataplatform.bronze import ingest_fixtures as run

    n = run(league, season)
    print(f"[green]✓[/green] {n} partidos crudos guardados en Bronze.")


@app.command()
def ingest_leagues():
    """[BRONZE] Cataloga todas las ligas/competiciones (ids para todo lo demás)."""
    from worldcup.dataplatform.api_client import ApiFootballClient
    from worldcup.dataplatform.bronze import ingest_leagues as run

    with ApiFootballClient() as client:
        n = run(client)
    print(f"[green]✓[/green] catálogo de ligas guardado ({n} filas).")


@app.command()
def backfill(
    league: int = typer.Option(..., help="id de liga/competición de api-football"),
    season: int = typer.Option(..., help="año de la temporada"),
    include: str = typer.Option(
        "statistics,events,lineups,players,odds",
        help="endpoints de detalle por partido (coma-separados)",
    ),
    force: bool = typer.Option(False, help="re-ingiere aunque ya exista (gasta cuota)"),
):
    """[BRONZE] Backfill COMPLETO y reanudable de una liga/temporada.

    Trae equipos + clasificación + partidos + todo el detalle por partido.
    Si se corta, vuelve a ejecutarlo y retoma donde quedó sin repetir llamadas.
    """
    from worldcup.dataplatform.bronze import backfill_season

    parts = [p.strip() for p in include.split(",") if p.strip()]
    print(f"[bold]Backfill[/bold] league={league} season={season} → {parts}")
    summary = backfill_season(league, season, include=parts, force=force)
    print("[green]✓[/green] backfill terminado:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


@app.command()
def seed(
    scope: str = typer.Option("national", help="national | clubs | all"),
    season_min: int = typer.Option(2018, help="temporada más antigua"),
    season_max: int = typer.Option(2026, help="temporada más reciente"),
    endpoints: str = typer.Option("statistics,events", help="detalle por partido (coma-sep)"),
    max_calls: int = typer.Option(7000, help="tope de llamadas para esta corrida (< tu cuota diaria)"),
    detail: bool = typer.Option(True, help="--no-detail = solo resultados, sin detalle caro"),
):
    """[BRONZE] Ingesta GUIADA: resuelve ligas por nombre y baja todo con presupuesto.

    Decide por ti qué competiciones traer para el Mundial 2026. Reanudable y con tope
    de llamadas: si llegas a tu cuota diaria, se detiene y mañana retomas.

    Ejemplos:
      wc seed --no-detail                      # rápido: todos los resultados (selecciones)
      wc seed --scope national --max-calls 7000
      wc seed --scope all --season-min 2020 --max-calls 7000
    """
    from worldcup.dataplatform.seed import run_seed

    parts = [p.strip() for p in endpoints.split(",") if p.strip()]
    summary = run_seed(
        scope=scope,
        seasons=(season_min, season_max),
        endpoints=parts,
        max_calls=max_calls,
        do_detail=detail,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


@app.command()
def build_silver():
    """[SILVER] Normaliza el crudo -> tablas matches, statistics, events, player stats."""
    from worldcup.dataplatform.silver import build_all

    counts = build_all()
    print("[green]✓[/green] tablas Silver construidas:")
    print(json.dumps(counts, indent=2, ensure_ascii=False))


@app.command()
def build_features():
    """[GOLD] Calcula TODAS las tablas de features: 1X2, tarjetas, córners, tiros por jugador."""
    from worldcup.dataplatform import features as F

    counts = {
        "match_features": F.build_match_features(),
        "card_features": F.build_card_features(),
        "corner_features": F.build_corner_features(),
        "player_shot_features": F.build_player_shot_features(),
    }
    print("[green]✓[/green] tablas Gold:")
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    if counts["card_features"] == 0 or counts["corner_features"] == 0:
        print("[dim]Tarjetas/córners vacíos: baja el detalle (wc seed --scope national).[/dim]")
    if counts["player_shot_features"] == 0:
        print("[dim]Tiros por jugador vacío: añade el endpoint players "
              "(wc seed --scope national --endpoints players).[/dim]")


@app.command()
def train(test_frac: float = 0.2):
    """[ML] Entrena el modelo 1X2 con validación temporal."""
    from worldcup.predictor.train import train_1x2

    metrics = train_1x2(test_frac=test_frac)
    print("[bold]Métricas de validación:[/bold]")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


@app.command()
def train_cards(test_frac: float = 0.2):
    """[ML] Entrena el modelo de tarjetas (Poisson + árbitro). Requiere detalle descargado."""
    from worldcup.predictor.cards import train_cards as run

    print(json.dumps(run(test_frac=test_frac), indent=2, ensure_ascii=False))


@app.command()
def train_corners(test_frac: float = 0.2):
    """[ML] Entrena el modelo de córners (Poisson). Requiere detalle descargado."""
    from worldcup.predictor.corners import train_corners as run

    print(json.dumps(run(test_frac=test_frac), indent=2, ensure_ascii=False))


@app.command()
def train_player_shots(test_frac: float = 0.2):
    """[ML] Entrena tiros a puerta por jugador (Poisson). Requiere endpoint players."""
    from worldcup.predictor.player_shots import train_player_shots as run

    print(json.dumps(run(test_frac=test_frac), indent=2, ensure_ascii=False))


@app.command()
def predict(
    home: int = typer.Option(...),
    away: int = typer.Option(...),
    referee: str = typer.Option(None, help="nombre del árbitro (mejora la predicción de tarjetas)"),
    neutral: bool = typer.Option(False, help="--neutral para sede neutral (Mundial)"),
):
    """[ML] Predice un partido: 1X2, goles, tarjetas y córners (por team_id)."""
    from worldcup.predictor.predict import predict_match

    print(json.dumps(predict_match(home, away, referee=referee, neutral=neutral),
                     indent=2, ensure_ascii=False))


@app.command()
def check():
    """[CALIDAD] Valida las tablas Silver/Gold (puertas de calidad del pipeline)."""
    from worldcup import quality

    results = quality.run_checks()
    for r in results:
        icon = "✓" if r.ok else ("✗" if r.critical else "⚠")
        color = "green" if r.ok else ("red" if r.critical else "yellow")
        print(f"[{color}]{icon}[/{color}] {r.name} [dim]({r.detail})[/dim]")
    ok = quality.gate(results)
    print("[green]✓ Calidad OK[/green]" if ok else "[red]✗ Fallos críticos[/red]")
    raise typer.Exit(code=0 if ok else 1)


@app.command()
def pipeline(no_train: bool = typer.Option(False, help="solo construir datos, sin entrenar")):
    """[ORQUESTACIÓN] Corre el pipeline: Silver→calidad→Gold→calidad→modelos."""
    from worldcup.pipeline import run_pipeline

    summary = run_pipeline(train_models=not no_train)
    print("[bold]Resumen:[/bold]")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


@app.command()
def predict_player(
    player: int = typer.Option(..., help="player_id de api-football"),
    away: bool = typer.Option(False, help="--away si juega de visitante"),
):
    """[ML] Predice los tiros a puerta de un jugador en su próximo partido."""
    from worldcup.predictor.predict import predict_player_shots

    print(json.dumps(predict_player_shots(player, is_home=not away), indent=2, ensure_ascii=False))


@app.command()
def dashboard(port: int = typer.Option(8501, help="puerto del dashboard")):
    """[VIZ] Lanza el dashboard Streamlit (predicciones con gráficas + exploración)."""
    import subprocess

    import worldcup.dashboard as dash

    subprocess.run(
        ["streamlit", "run", dash.__file__, "--server.port", str(port)], check=False
    )


if __name__ == "__main__":
    app()
