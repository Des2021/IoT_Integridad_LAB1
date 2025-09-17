# app.py
"""
Mini-Lab IoT: Integridad & Capas
Streamlit app lista para desplegar en Streamlit Community Cloud.
- Usa matplotlib (no seaborn)
- HMAC key desde st.secrets["hmac_key"]
"""

import streamlit as st
import numpy as np
import time
import hashlib
import hmac
import io
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime

st.set_page_config(page_title="Mini-Lab IoT — Integridad & Capas", layout="wide")

# ---------- Helper functions ----------
def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def hmac_sha256(msg: str, key: str) -> str:
    return hmac.new(key.encode(), msg.encode(), hashlib.sha256).hexdigest()

# Simulador de sensor
def sensor_temp(base=25.0, noise=0.3, drift=0.002, t=0):
    """Base + ruido gaussiano + deriva lenta."""
    return base + np.random.normal(0, noise) + drift * t

# Canal de red con pérdida/latencia y manipulación opcional
class Canal:
    def __init__(self, loss_prob=0.05, min_ms=20, max_ms=120, tamper=False, tamper_bias=10.0):
        self.loss_prob = loss_prob
        self.min_ms = min_ms
        self.max_ms = max_ms
        self.tamper = tamper
        self.tamper_bias = tamper_bias

    def enviar(self, valor):
        # pérdida
        if np.random.rand() < self.loss_prob:
            return None, None
        # latencia
        lat = np.random.uniform(self.min_ms, self.max_ms)
        # manipulación
        v = valor + self.tamper_bias if self.tamper else valor
        return v, lat

# ---------- UI ----------
st.title("📡 Mini-Lab IoT — Integridad & Capas (Streamlit)")

col1, col2 = st.columns([2,1])

with col2:
    st.header("Control")
    # Duración en samples (evita loops largos en cloud)
    n_samples = st.slider("Número de muestras (simulación)", min_value=10, max_value=500, value=80, step=10)
    sample_interval = st.slider("Intervalo entre muestras (s)", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
    loss_prob = st.slider("Prob. pérdida de paquete", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
    tamper_toggle = st.radio("Tampering", options=["Sin manipulación", "Manipulado"], index=0)
    tamper_bias = st.number_input("Sesgo de manipulación (°C)", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
    integrity = st.selectbox("Verificación de integridad", options=["Sin verificación", "SHA256", "HMAC"])
    seed = st.number_input("Semilla aleatoria (para reproducibilidad)", value=7, step=1)
    start_btn = st.button("▶ Ejecutar simulación")
    st.info("La clave HMAC debe configurarse en Streamlit Secrets con la clave 'hmac_key' (ver README).", icon="ℹ️")

with col1:
    st.header("Visualización")
    chart_placeholder = st.empty()
    summary_placeholder = st.empty()

# ---------- Check secrets ----------
# We'll fallback to a local generated key if st.secrets not set (but recommend using Streamlit Secrets)
hmac_key = None
if "hmac_key" in st.secrets:
    hmac_key = st.secrets["hmac_key"]
else:
    # mostrar advertencia clara si no está configurado
    st.warning("⚠️ Streamlit Secrets no contiene 'hmac_key'. La app generará una clave temporal (NO para producción).", icon="⚠️")
    # generar clave temporal reproducible a partir de seed
    hmac_key = hashlib.sha256(f"temp_key_{seed}".encode()).hexdigest()[:32]

# ---------- Ejecución de la simulación ----------
if start_btn:
    np.random.seed(seed)
    canal = Canal(loss_prob=loss_prob, tamper=(tamper_toggle=="Manipulado"), tamper_bias=tamper_bias)
    timestamps = []
    rx_vals = []
    verdicts = []
    sigs = []
    latencias = []

    cnt = 0
    # Buffers para plot
    xs = []
    ys = []
    marks = []

    for i in range(n_samples):
        t = i
        raw = sensor_temp(t=t)
        msg = f"{raw:.2f}"
        sig = None

        if integrity == "SHA256":
            sig = sha256_str(msg)
        elif integrity == "HMAC":
            sig = hmac_sha256(msg, hmac_key)

        rx, lat = canal.enviar(float(msg))
        if rx is None:
            # paquete perdido -> registrar como NaN
            timestamps.append(datetime.now())
            rx_vals.append(np.nan)
            verdicts.append(False)
            sigs.append(sig)
            latencias.append(None)
            # pequeño delay y continuar
            time.sleep(sample_interval)
            continue

        # verificación en "plataforma"
        ok = True
        if integrity == "SHA256":
            ok = (sha256_str(f"{rx:.2f}") == sig)
        elif integrity == "HMAC":
            ok = (hmac_sha256(f"{rx:.2f}", hmac_key) == sig)

        timestamps.append(datetime.now())
        rx_vals.append(rx)
        verdicts.append(ok)
        sigs.append(sig)
        latencias.append(lat)

        # para el plot en tiempo real
        xs.append(i)
        ys.append(rx)
        marks.append(0 if ok else 1)  # 1 indica fallo

        # Dibujar con matplotlib (cumple la regla: matplotlib, no seaborn)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(xs, ys, marker='o', linewidth=1, label='Temperatura recibida (°C)')
        # marcar fallos
        bad_x = [x for x,m in zip(xs,marks) if m==1]
        bad_y = [y for y,m in zip(ys,marks) if m==1]
        if bad_x:
            ax.scatter(bad_x, bad_y, s=80, edgecolor='k', c='r', label='Fallo integridad')
            ax.legend()
        ax.set_xlabel("Muestras")
        ax.set_ylabel("°C")
        ax.set_title("Temperatura recibida (simulación)")
        ax.grid(True)

        chart_placeholder.pyplot(fig)
        # summary
        total = len([v for v in verdicts if v is not None])
        malos = sum(1 for v in verdicts if v is False)
        summary_placeholder.markdown(f"**Muestras:** {i+1}  |  **Fallos integridad detectados:** {malos}  |  **Tampering:** {tamper_toggle}  |  **Verificación:** {integrity}")
        time.sleep(sample_interval)

    # al final, permitir descargar datos y figura
    df = pd.DataFrame({
        "timestamp": timestamps,
        "rx_value": rx_vals,
        "integrity_ok": verdicts,
        "signature": sigs,
        "latency_ms": latencias
    })

    st.success("Simulación completada ✅")
    st.markdown("### Resultado (tabla) y descargas")
    st.dataframe(df.head(200))

    # Descargar CSV
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("📥 Descargar CSV", csv_bytes, file_name="sim_results.csv", mime="text/csv")

    # Descargar imagen final
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    st.download_button("📷 Descargar imagen del gráfico", data=buf, file_name="grafico_final.png", mime="image/png")

    # Comentario pedagógico (mentoring)
    st.markdown("---")
    st.header("Reflexión (mentor)")
    st.markdown("""
    - **SHA256**: ilustra hashing básico. Si el atacante puede recalcular el hash (p. ej. tiene control del flujo), no impide tampering.
    - **HMAC**: exige una clave secreta compartida; sin la clave, un atacante no puede recomputar la firma -> detecta manipulación en tránsito.
    - **Limitación**: HMAC es centralizado (requiere gestión de claves). Blockchain aparece como solución cuando necesitas trazabilidad inmutable y verificación distribuida.
    - **Trade-offs**: coste, latencia, complejidad de integración (ver caso InduWare: refuerzos vs Blockchain privado vs híbrido).
    """)

    st.markdown("------")
    st.caption("App educativa: evita exponer la clave HMAC en código público; usa Streamlit Secrets para producción.")
