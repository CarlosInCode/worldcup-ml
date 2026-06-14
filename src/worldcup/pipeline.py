"""Orquestador del pipeline: ejecuta el grafo Silver -> calidad -> Gold -> calidad -> modelos.

Mejor práctica: las etapas tienen dependencias explícitas y PUERTAS DE CALIDAD entre ellas.
Si una validación crítica falla, el pipeline se detiene en vez de entrenar sobre datos malos.

Es un orquestador ligero (sin servidor). El siguiente paso de escalado sería migrarlo a
Dagster (assets = estas mismas funciones) cuando el entorno lo soporte; el diseño ya está
pensado para esa migración: cada etapa es una función pura con dependencias declaradas.
"""

from __future__ import annotations

from rich import print

from worldcup import quality


def _run_quality_gate(label: str) -> bool:
    results = quality.run_checks()
    failed_critical = [r for r in results if r.critical and not r.ok]
    warns = [r for r in results if not r.critical and not r.ok]
    for r in results:
        icon = "[green]✓[/green]" if r.ok else ("[red]✗[/red]" if r.critical else "[yellow]⚠[/yellow]")
        print(f"   {icon} {r.name} ({r.detail})")
    if failed_critical:
        print(f"[red]✗ Puerta de calidad '{label}' BLOQUEADA: {len(failed_critical)} fallos críticos.[/red]")
        return False
    if warns:
        print(f"[yellow]⚠ {len(warns)} avisos no críticos en '{label}' (continúo).[/yellow]")
    return True


def run_pipeline(train_models: bool = True) -> dict:
    """Corre el pipeline completo desde Bronze ya ingerido. Devuelve un resumen.

    Etapas: build Silver -> gate -> build Gold -> gate -> entrenar todos los modelos posibles.
    No ingiere (eso es `wc seed`); asume que Bronze ya tiene datos.
    """
    from worldcup.dataplatform import features as F
    from worldcup.dataplatform import silver

    summary: dict = {"stages": {}, "models": {}, "halted": None}

    print("[bold cyan]1/4 · Silver[/bold cyan] — normalizando crudo")
    summary["stages"]["silver"] = silver.build_all()
    print(f"   {summary['stages']['silver']}")

    print("[bold]2/4 · Puerta de calidad (Silver)[/bold]")
    if not _run_quality_gate("silver"):
        summary["halted"] = "quality_gate_silver"
        return summary

    print("[bold cyan]3/4 · Gold[/bold cyan] — calculando features")
    summary["stages"]["gold"] = {
        "match_features": F.build_match_features(),
        "card_features": F.build_card_features(),
        "corner_features": F.build_corner_features(),
        "player_shot_features": F.build_player_shot_features(),
    }
    print(f"   {summary['stages']['gold']}")

    print("[bold]Puerta de calidad (Gold)[/bold]")
    if not _run_quality_gate("gold"):
        summary["halted"] = "quality_gate_gold"
        return summary

    if not train_models:
        return summary

    print("[bold cyan]4/4 · Modelos[/bold cyan] — entrenando lo que tenga datos suficientes")
    _train_safe(summary, "1x2", lambda: __import__("worldcup.predictor.train", fromlist=["train_1x2"]).train_1x2())
    _train_safe(summary, "cards", lambda: __import__("worldcup.predictor.cards", fromlist=["train_cards"]).train_cards())
    _train_safe(summary, "corners", lambda: __import__("worldcup.predictor.corners", fromlist=["train_corners"]).train_corners())
    _train_safe(summary, "player_shots", lambda: __import__("worldcup.predictor.player_shots", fromlist=["train_player_shots"]).train_player_shots())
    return summary


def _train_safe(summary: dict, name: str, fn) -> None:
    """Entrena un modelo; si no hay datos suficientes, lo registra como omitido (no rompe)."""
    try:
        summary["models"][name] = fn()
        print(f"   [green]✓[/green] {name} entrenado")
    except Exception as e:
        summary["models"][name] = {"skipped": str(e).split(chr(10))[0]}
        print(f"   [yellow]⊘[/yellow] {name} omitido: {summary['models'][name]['skipped']}")
