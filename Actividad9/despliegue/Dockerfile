# VOREKO IA · Actividad 9 · Imagen del microservicio de inferencia optimizado

FROM python:3.12-slim
LABEL org.opencontainers.image.title="voreko-ia-inferencia" \
      org.opencontainers.image.version="1.1.0" \
      org.opencontainers.image.authors="José Ignacio Castillo Camacho" \
      org.opencontainers.image.source="https://github.com/Ignacio-Castillo-C/voreko-ia-incidencias"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080 \
    LIMITE_LOTE=500 \
    VOREKO_CACHE=5000 \
    VOREKO_CORS="https://app.voreko.mx" \
    VOREKO_DIR_MODELO_OPT=/app/Actividad9/modelo/optimizado
RUN python -m venv /opt/venv && useradd --create-home --uid 10001 voreko
COPY Actividad9/api/requirements-api-optimizada.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt
WORKDIR /app
COPY --chown=voreko:voreko FaseII/api/esquemas.py FaseII/api/esquemas.py
COPY --chown=voreko:voreko Actividad9/api/main_optimizada.py Actividad9/api/servicio_optimizado.py Actividad9/api/
COPY --chown=voreko:voreko Actividad9/modelo/optimizado Actividad9/modelo/optimizado
USER voreko
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn main_optimizada:app --app-dir Actividad9/api --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips '*'"]
