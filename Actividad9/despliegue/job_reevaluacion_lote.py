# VOREKO IA · Actividad 9 · Trabajo nocturno de reevaluación por lotes
# José Ignacio Castillo Camacho · AL07161745

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RAIZ_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ_REPO / "Actividad9" / "api"))
from servicio_optimizado import ServicioOptimizado


def main(entrada: Path, salida: Path, directorio_modelo: Path, bloque: int):
    servicio = ServicioOptimizado(directorio_modelo).cargar()
    with entrada.open(encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    t0 = time.perf_counter()
    columnas = {c: np.array([float(r[c]) for r in filas]) for c in servicio.columnas}
    prob = np.concatenate([servicio.modelo.run([servicio._salida], {c: columnas[c][k:k + bloque].astype(servicio._tipos[c]).reshape(-1, 1)
                                                                     for c in servicio.columnas})[0][:, 1] for k in range(0, len(filas), bloque)])
    segundos = time.perf_counter() - t0
    fecha = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with salida.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id_proyecto", "probabilidad", "nivel", "version_modelo", "fecha_evaluacion"])
        for r, p in zip(filas, prob):
            w.writerow([r.get("id_proyecto", ""), round(float(p), 4), servicio.nivel(float(p)), servicio.version, fecha])
    niveles = {n: int(sum(servicio.nivel(float(p)) == n for p in prob)) for n in ("verde", "amarillo", "rojo")}
    print(json.dumps({"obras": len(filas), "segundos": round(segundos, 3), "obras_por_segundo": round(len(filas) / segundos, 1),
                      "por_nivel": niveles, "version_modelo": servicio.version}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--entrada", type=Path, required=True)
    p.add_argument("--salida", type=Path, required=True)
    p.add_argument("--modelo", type=Path, default=RAIZ_REPO / "Actividad9" / "modelo" / "optimizado")
    p.add_argument("--bloque", type=int, default=10_000)
    a = p.parse_args()
    main(a.entrada, a.salida, a.modelo, a.bloque)
