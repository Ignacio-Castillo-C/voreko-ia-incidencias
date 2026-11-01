# VOREKO IA · Fase II · Prueba de humo del contenedor en ejecución

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
SALIDA = Path(__file__).resolve().parent / "resultados" / "prueba_contenedor.json"


# --- Cliente HTTP mínimo ---

def pedir(metodo: str, ruta: str, cuerpo=None, crudo: str | None = None):
    datos = crudo.encode() if crudo is not None else (json.dumps(cuerpo).encode() if cuerpo is not None else None)
    req = urllib.request.Request(URL + ruta, data=datos, method=metodo, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            texto, estado = r.read().decode(), r.status
    except urllib.error.HTTPError as e:
        texto, estado = e.read().decode(), e.code
    ms = (time.perf_counter() - t0) * 1000
    try:
        return estado, json.loads(texto), ms
    except json.JSONDecodeError:
        return estado, texto, ms


# --- Casos ---

def main():
    casos = []

    def caso(nombre, ok, detalle=""):
        casos.append({"caso": nombre, "resultado": "OK" if ok else "FALLA", "detalle": detalle})
        print(f"{'OK   ' if ok else 'FALLA'} {nombre} {detalle}")

    e, s, ms = pedir("GET", "/api/salud")
    caso("Salud del servicio", e == 200 and s["estado"] == "ok", f"{s.get('version_modelo')} · {ms:.0f} ms")
    e, m, _ = pedir("GET", "/api/modelo")
    caso("Información del modelo", e == 200 and m["umbrales"]["rojo"] == 0.39, f"umbral rojo {m['umbrales']['rojo']}")
    e, obras, _ = pedir("GET", "/api/obras-ejemplo?completo=true")
    caso("Obras de ejemplo", e == 200 and len(obras) == 75, f"{len(obras)} obras")
    o = obras[0]
    e, r, ms = pedir("POST", "/api/predecir", {"id_proyecto": o["id_proyecto"], "variables": o["variables"]})
    caso("Predicción individual", e == 200 and abs(r["probabilidad"] - round(o["probabilidad_referencia"], 4)) < 1e-9,
         f"{o['id_proyecto']}: {r['probabilidad_pct']} % · {r['nivel']} · {ms:.0f} ms")
    e, lote, ms = pedir("POST", "/api/predecir/lote", {"obras": [{"id_proyecto": x["id_proyecto"], "variables": x["variables"],
                                                                 "incluir_factores": False} for x in obras]})
    res = {x["id_proyecto"]: x["resultado"] for x in lote["resultados"]}
    vp = sum(1 for x in obras if res[x["id_proyecto"]]["alerta"] and x["riesgo_real"])
    fp = sum(1 for x in obras if res[x["id_proyecto"]]["alerta"] and not x["riesgo_real"])
    fn = sum(1 for x in obras if not res[x["id_proyecto"]]["alerta"] and x["riesgo_real"])
    caso("Lote de 75 obras", e == 200 and (vp, fp, fn) == (14, 11, 12),
         f"{lote['por_nivel']} · VP={vp} FP={fp} FN={fn} · recall {vp / (vp + fn):.3f} · precisión {vp / (vp + fp):.3f} · {ms:.0f} ms")
    e, err, _ = pedir("POST", "/api/predecir", {"variables": {**o["variables"], "area_construida_m2": -10}})
    caso("Rechazo de área negativa", e == 422, json.dumps(err.get("errores"), ensure_ascii=False))
    e, err, _ = pedir("POST", "/api/predecir", crudo="{malo")
    caso("Rechazo de JSON mal formado", e == 422)
    e, _, _ = pedir("POST", "/api/predecir/lote", {"obras": []})
    caso("Rechazo de lote vacío", e == 422)
    e, html, _ = pedir("GET", "/")
    caso("Frontend disponible", e == 200 and "VOREKO" in str(html))
    e, _, _ = pedir("GET", "/docs")
    caso("Documentación Swagger disponible", e == 200)

    tiempos = [pedir("POST", "/api/predecir", {"variables": o["variables"]})[2] for _ in range(100)]
    p50, p95 = statistics.median(tiempos), sorted(tiempos)[94]
    caso("Latencia de 100 predicciones con factores", p95 < 1000, f"mediana {p50:.0f} ms · p95 {p95:.0f} ms")

    resumen = {"url": URL, "fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "total": len(casos),
               "aprobados": sum(c["resultado"] == "OK" for c in casos), "latencia_ms": {"mediana": round(p50, 1), "p95": round(p95, 1)},
               "casos": casos}
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{resumen['aprobados']}/{resumen['total']} casos aprobados")
    sys.exit(0 if resumen["aprobados"] == resumen["total"] else 1)


if __name__ == "__main__":
    main()
