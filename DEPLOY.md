# 🚀 Publicar el dashboard online (gratis)

El dashboard es **solo lectura**: lee `data/warehouse.duckdb` y `models/`, **no llama a
api-football**. Por eso NO necesitas exponer tu API key y el despliegue es sencillo.

## Qué se sube (≈15 MB) y qué NO

| Sube ✅ | NO subas ❌ |
|---------|------------|
| Código (`src/`, `streamlit_app.py`) | `.env` (tu API key) |
| `requirements.txt`, `packages.txt` | `data/bronze/` (crudo, pesado e innecesario en runtime) |
| `data/warehouse.duckdb` (tablas Gold/Silver) | `mlflow.db`, `mlruns/` |
| `models/` (modelos entrenados) | |

> Estos artefactos están en `.gitignore` (son "build artifacts"). Para el deploy se
> fuerzan con `git add -f` (ver abajo). Así el repo queda limpio por defecto.

---

## Opción A — Streamlit Community Cloud (recomendada)

Gratis, hecho por Streamlit, despliega directo desde GitHub.

```bash
# 1. Inicializa git y haz el primer commit
cd /Users/carlos/Documents/machine_learning
git init && git add .
git add -f data/warehouse.duckdb models/        # fuerza los artefactos ignorados
git commit -m "World Cup ML dashboard"

# 2. Crea un repo en GitHub y súbelo
#    (con la web de GitHub o:  gh repo create worldcup-ml --public --source=. --push)
git remote add origin https://github.com/<tu-usuario>/worldcup-ml.git
git push -u origin main
```

3. Entra a **https://share.streamlit.io** → inicia sesión con GitHub → **New app**.
4. Elige tu repo, rama `main`, y **Main file path = `streamlit_app.py`**.
5. **Deploy**. En ~2-3 min tendrás una URL pública tipo
   `https://<tu-app>.streamlit.app` para compartir.

`requirements.txt` (deps con versión fija) y `packages.txt` (`libgomp1`, que XGBoost
necesita en Linux) ya están listos: Streamlit Cloud los detecta solo.

---

## Opción B — Hugging Face Spaces (alternativa)

Buena si quieres archivos grandes (usa git-lfs) o un perfil de comunidad ML.

1. Crea un **Space** en https://huggingface.co/new-space → SDK **Streamlit**.
2. Sube los mismos archivos (incluye `streamlit_app.py`, `requirements.txt`,
   `data/warehouse.duckdb`, `models/`). En el Space, renombra/duplica
   `streamlit_app.py` como `app.py` (HF busca `app.py` por defecto) o ajusta la config.

---

## Para refrescar los datos en producción

El `warehouse.duckdb` desplegado es una **foto** del momento del commit. Cuando bajes más
datos y reentrenes localmente (`wc seed ... && wc pipeline`), vuelve a subir:

```bash
git add -f data/warehouse.duckdb models/
git commit -m "Actualiza datos y modelos"
git push           # Streamlit Cloud redespliega solo
```

*(Patrón más avanzado para después: subir el .duckdb a almacenamiento en la nube y que la
app lo descargue al arrancar, en vez de versionarlo en git.)*

---

## ⚠️ Aviso legal

Revisa los **términos de api-football**: suelen permitir usar los datos pero **restringir
su redistribución masiva**. El dashboard muestra *agregados y predicciones* (bajo riesgo),
no volcados crudos del API. Si tienes dudas, mantén el Space privado o limita lo que expones.
