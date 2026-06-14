"""Cliente para api-football (API-Sports / RapidAPI).

Responsabilidades:
  - Autenticación (soporta tanto el endpoint directo de API-Sports como RapidAPI).
  - Rate limiting: nunca superar las llamadas/minuto de tu plan.
  - Reintentos con backoff ante errores transitorios (429, 5xx, timeouts).
  - Paginación automática (api-football pagina con `paging.current`/`paging.total`).

NO transforma datos: devuelve el JSON tal cual. La limpieza ocurre en la capa Silver.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from worldcup.common.config import settings


class _RateLimiter:
    """Limitador de ventana deslizante: máx. N llamadas en los últimos 60s.

    Es thread-safe para que puedas paralelizar la ingesta más adelante sin pasarte
    del límite de tu plan (y ahorrarte baneos temporales del API).
    """

    def __init__(self, max_per_min: int):
        self.max_per_min = max_per_min
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            # Descarta timestamps de hace más de 60s.
            while self._calls and now - self._calls[0] > 60:
                self._calls.popleft()
            if len(self._calls) >= self.max_per_min:
                sleep_for = 60 - (now - self._calls[0]) + 0.05
                time.sleep(max(sleep_for, 0))
            self._calls.append(time.monotonic())


class ApiFootballError(Exception):
    pass


class ApiFootballClient:
    def __init__(self) -> None:
        if not settings.api_football_key:
            raise ApiFootballError(
                "Falta API_FOOTBALL_KEY. Copia .env.example a .env y rellénala."
            )
        self._limiter = _RateLimiter(settings.api_football_rate_limit_per_min)

        if settings.uses_rapidapi:
            headers = {
                "x-rapidapi-key": settings.api_football_key,
                "x-rapidapi-host": settings.api_football_rapidapi_host or "",
            }
        else:
            headers = {"x-apisports-key": settings.api_football_key}

        self._client = httpx.Client(
            base_url=settings.api_football_base_url,
            headers=headers,
            timeout=30.0,
        )
        # Cuántas llamadas reales hemos hecho al API (para respetar la cuota diaria).
        self.calls_made = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ApiFootballClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, ApiFootballError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get_page(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        self._limiter.acquire()
        self.calls_made += 1  # cada intento cuenta contra tu cuota diaria
        resp = self._client.get(f"/{endpoint.lstrip('/')}", params=params)

        # 429 = te pasaste del rate limit -> reintentar con backoff.
        if resp.status_code == 429:
            raise ApiFootballError("429 rate limit; reintentando con backoff")
        resp.raise_for_status()
        data = resp.json()

        # api-football mete los errores de negocio dentro del cuerpo, no en el status.
        errors = data.get("errors")
        if errors:
            # Algunos errores (p.ej. cuota agotada del día) no se arreglan reintentando.
            raise ApiFootballError(f"API devolvió errores en {endpoint}: {errors}")
        return data

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Llama un endpoint y devuelve TODAS las páginas concatenadas (la lista `response`).

            client.get("fixtures", {"league": 39, "season": 2023})
        """
        base_params = dict(params or {})
        # 1ª petición SIN 'page': algunos endpoints (leagues, etc.) rechazan ese campo.
        data = self._get_page(endpoint, base_params)
        out: list[dict[str, Any]] = list(data.get("response", []))

        paging = data.get("paging", {}) or {}
        total = int(paging.get("total", 1) or 1)
        # Solo paginamos si de verdad hay más de una página.
        for page in range(2, total + 1):
            data = self._get_page(endpoint, {**base_params, "page": page})
            out.extend(data.get("response", []))
        return out
