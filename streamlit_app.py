"""Punto de entrada para el hosting (Streamlit Community Cloud / Hugging Face Spaces).

El paquete vive en src/worldcup (layout src), así que añadimos src/ al path e importamos
el dashboard, cuyas llamadas a Streamlit se ejecutan al importarse.

En Streamlit Cloud configura el "Main file path" como:  streamlit_app.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import worldcup.dashboard  # noqa: E402,F401  (ejecuta la app al importar)
