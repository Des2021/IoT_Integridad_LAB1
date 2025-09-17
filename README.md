# IoT_Integridad_LAB1

# Mini-Lab IoT — Integridad & Capas (Streamlit)

App educativa para simular sensor → canal → verificación (SHA256 / HMAC).
Listo para desplegar en Streamlit Community Cloud desde GitHub (solo web).

## Archivos
- `app.py` : app principal
- `requirements.txt` : dependencias

## Requisitos
- Cuenta en GitHub
- Cuenta en Streamlit (Streamlit Community Cloud)

## Pasos para desplegar (sin usar terminal, solo navegador)

### 1) Crear repo en GitHub
1. Entra a GitHub, pulsa **New repository**.
2. Ponle un nombre (p.ej. `mini-lab-iot-streamlit`) y crea el repo (público o privado).

### 2) Añadir archivos por la web
1. Dentro del repo, pulsa **Add file → Create new file**.
2. Crea `requirements.txt`, pega el contenido indicado y **Commit**.
3. Repite para `app.py` y `README.md`. Commit cada uno.

### 3) Configurar Streamlit Secrets (clave HMAC)
1. Ve a [streamlit.io/cloud](https://streamlit.io/cloud) y conecta tu cuenta de GitHub si no está conectada.
2. Crea una nueva app → selecciona el repo y la rama.
3. Antes de lanzar, en **Advanced settings** > **Secrets** añade:
   ```toml
   hmac_key = "tu_clave_secreta_aqui"

https://iotintegridadlab1-gkadfc7pzpjz7kupvtak4z.streamlit.app
