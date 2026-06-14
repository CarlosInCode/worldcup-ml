"""Configuración central. Se carga desde variables de entorno / .env.

Punto único de verdad para rutas y credenciales. Importa `settings` donde lo necesites:

    from worldcup.common.config import settings
    print(settings.bronze_dir)
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- api-football ---
    api_football_key: str = Field(default="", alias="API_FOOTBALL_KEY")
    api_football_base_url: str = Field(
        default="https://v3.football.api-sports.io", alias="API_FOOTBALL_BASE_URL"
    )
    api_football_rapidapi_host: str | None = Field(
        default=None, alias="API_FOOTBALL_RAPIDAPI_HOST"
    )
    api_football_rate_limit_per_min: int = Field(
        default=300, alias="API_FOOTBALL_RATE_LIMIT_PER_MIN"
    )

    # --- Rutas (medallón) ---
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")

    @property
    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def duckdb_path(self) -> Path:
        """Warehouse analítico embebido donde viven las tablas Silver/Gold."""
        return self.data_dir / "warehouse.duckdb"

    @property
    def uses_rapidapi(self) -> bool:
        return self.api_football_rapidapi_host is not None


settings = Settings()
