.PHONY: setup demo lint test clean

# Instala todo (base + ml + viz + dev) en un entorno virtual gestionado por uv.
setup:
	uv sync --extra ml --extra viz --extra dev

# Pipeline de demostración end-to-end (requiere .env con tu API key).
# Ingiere varias temporadas de la Premier (league 39) para tener datos suficientes.
demo:
	uv run wc ingest-fixtures --league 39 --season 2021
	uv run wc ingest-fixtures --league 39 --season 2022
	uv run wc ingest-fixtures --league 39 --season 2023
	uv run wc build-silver
	uv run wc build-features
	uv run wc train

lint:
	uv run ruff check src/

test:
	uv run pytest -q

clean:
	rm -rf data/*.duckdb data/*.duckdb.wal models/ mlruns/
