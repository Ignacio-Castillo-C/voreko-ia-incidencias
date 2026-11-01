# VOREKO IA · Fase II · Imagen del servicio de alertas de sobrecosto

ARG PYTHON_VERSION=3.12

# --- Etapa 1: entrenamiento y verificación del modelo ---
FROM python:${PYTHON_VERSION}-slim AS entrenamiento
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PATH="/opt/venv/bin:$PATH"
RUN python -m venv /opt/venv
WORKDIR /repo
COPY FaseII/requirements-entrenamiento.txt FaseII/
RUN pip install -r FaseII/requirements-entrenamiento.txt
COPY Actividad5/fuentes Actividad5/fuentes
COPY Actividad5/datos/datos_limp Actividad5/datos/datos_limp
COPY Actividad6/fuentes/voreko_a6.py Actividad6/fuentes/
COPY Actividad6/datos Actividad6/datos
COPY FaseII/modelo/entrenar_modelo.py FaseII/modelo/
RUN python FaseII/modelo/entrenar_modelo.py --salida /artefactos

# --- Etapa 2: servicio (API + frontend) ---
FROM python:${PYTHON_VERSION}-slim AS servicio
LABEL org.opencontainers.image.title="voreko-ia" \
      org.opencontainers.image.description="API y frontend del sistema de alertas de sobrecosto de VOREKO IA" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.authors="José Ignacio Castillo Camacho" \
      org.opencontainers.image.source="https://github.com/Ignacio-Castillo-C/voreko-ia-incidencias"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    LIMITE_LOTE=500 \
    VOREKO_CORS="*" \
    VOREKO_DIR_MODELO=/app/modelo \
    VOREKO_DIR_FRONTEND=/app/frontend
RUN python -m venv /opt/venv && useradd --create-home --uid 10001 voreko
COPY FaseII/requirements-api.txt /tmp/requirements-api.txt
RUN pip install -r /tmp/requirements-api.txt && rm /tmp/requirements-api.txt
WORKDIR /app
COPY --chown=voreko:voreko FaseII/api ./api
COPY --chown=voreko:voreko FaseII/frontend ./frontend
COPY --from=entrenamiento --chown=voreko:voreko /artefactos ./modelo
USER voreko
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/salud' % os.environ.get('PORT', '8000'), timeout=4)"]
CMD ["sh", "-c", "exec uvicorn main:app --app-dir api --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips '*'"]
