"""Registro de modelos (champion/challenger).

Mejor práctica de MLOps: al reentrenar, NO reemplaces el modelo vigente a ciegas. Compara
el "retador" (nuevo) contra el "campeón" (actual) y promociona solo si mejora la métrica.
Así nunca empeoras el modelo en producción por un reentrenamiento con datos ruidosos.

Guarda el historial en models/registry.json.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

_REGISTRY = Path("models/registry.json")


def _load() -> dict:
    if _REGISTRY.exists():
        return json.loads(_REGISTRY.read_text())
    return {}


def _save(reg: dict) -> None:
    _REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY.write_text(json.dumps(reg, indent=2, ensure_ascii=False))


def promote(
    model_key: str,
    new_score: float,
    save_fn: Callable[[], None],
    *,
    higher_is_better: bool,
    metrics: dict | None = None,
) -> dict:
    """Decide si el retador reemplaza al campeón. Guarda el modelo solo si promociona.

    save_fn: función que persiste el modelo nuevo (se llama solo si promociona).
    Devuelve {promoted, new_score, previous_score}.
    """
    reg = _load()
    prev = reg.get(model_key, {}).get("score")
    better = (
        prev is None
        or (new_score > prev if higher_is_better else new_score < prev)
    )
    if better:
        save_fn()
        reg[model_key] = {"score": new_score, "metrics": metrics or {}}
        _save(reg)
    return {"promoted": better, "new_score": new_score, "previous_score": prev}
