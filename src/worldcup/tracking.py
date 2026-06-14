"""Tracking de experimentos con MLflow.

Mejor práctica de ML: registrar cada entrenamiento (parámetros + métricas + fecha) para
poder comparar versiones del modelo en el tiempo y no perder qué funcionó. Se guarda en
local en SQLite (mlflow.db). Visualízalo con:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
y abre http://localhost:5000.

Es tolerante a fallos: si MLflow no está instalado o algo falla, no rompe el entrenamiento.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def track(experiment: str, params: dict[str, Any], metrics: dict[str, Any]) -> bool:
    """Registra una corrida. Solo se loguean métricas numéricas. Devuelve True si se registró."""
    try:
        import mlflow
    except ImportError:
        return False
    try:
        # Backend SQLite (el file store quedó deprecado en MLflow 3.x).
        mlflow.set_tracking_uri(f"sqlite:///{Path('mlflow.db').resolve()}")
        mlflow.set_experiment(experiment)
        num_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        # Lo no numérico (p.ej. nombre de tabla) se guarda como parámetro/tag.
        str_params = {**params, **{k: v for k, v in metrics.items() if not isinstance(v, (int, float))}}
        with mlflow.start_run():
            mlflow.log_params(str_params)
            mlflow.log_metrics(num_metrics)
        return True
    except Exception:
        return False
