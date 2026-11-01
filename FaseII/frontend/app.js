const API = window.VOREKO_API || "";
const PRINCIPALES = {
  anio_inicio: "Año de inicio (calendario persa)",
  trimestre_inicio: "Trimestre de inicio",
  localidad: "Localidad (zona 1 a 20)",
  area_construida_m2: "Área construida",
  area_terreno_m2: "Área del terreno",
  costo_prelim_m2: "Costo preliminar por m²",
  costo_prelim_m2_base: "Costo preliminar por m² (año base)",
  precio_venta_inicial_m2: "Precio de venta inicial por m²",
};
const NIVELES = { verde: "Riesgo bajo", amarillo: "Riesgo intermedio", rojo: "Riesgo alto" };
const estado = { modelo: null, obras: [], obraActual: null };
const $ = (s) => document.querySelector(s);
const fmt = (x, d = 2) => Number(x).toLocaleString("es-MX", { maximumFractionDigits: d });

async function llamar(ruta, opciones = {}) {
  const r = await fetch(API + ruta, { headers: { "Content-Type": "application/json" }, ...opciones });
  const cuerpo = await r.json().catch(() => ({}));
  if (!r.ok) {
    const error = new Error(cuerpo.detalle || cuerpo.detail || `Error ${r.status}`);
    error.errores = cuerpo.errores || [];
    error.estado = r.status;
    throw error;
  }
  return cuerpo;
}

function mostrarEstado(ok, texto) {
  const e = $("#estado-api");
  e.classList.toggle("ok", ok);
  e.classList.toggle("mal", !ok);
  e.querySelector(".texto").textContent = texto;
}

async function revisarSalud() {
  try {
    const s = await llamar("/api/salud");
    mostrarEstado(true, `API en línea · ${s.version_modelo}`);
    return true;
  } catch (e) {
    mostrarEstado(false, "API sin respuesta");
    return false;
  }
}

function crearCampo(v) {
  const etiqueta = PRINCIPALES[v.nombre] || v.descripcion;
  const caja = document.createElement("label");
  caja.className = "campo";
  caja.title = `${v.descripcion} · rango de entrenamiento ${fmt(v.minimo, 4)} a ${fmt(v.maximo, 4)}`;
  let control;
  if (v.tipo === "categórica") {
    control = document.createElement("select");
    v.valores_permitidos.forEach((x) => control.add(new Option(String(x), String(x))));
  } else {
    control = document.createElement("input");
    control.type = "number";
    control.step = "any";
    control.required = true;
  }
  control.name = v.nombre;
  control.addEventListener("input", () => marcarCambio(control));
  caja.append(etiqueta, control);
  if (v.unidad) {
    const u = document.createElement("span");
    u.className = "unidad";
    u.textContent = v.unidad;
    caja.append(u);
  }
  return caja;
}

function marcarCambio(control) {
  if (!estado.obraActual) return;
  const original = estado.obraActual.variables[control.name];
  control.parentElement.classList.toggle("modificado", Number(control.value) !== Number(original));
}

function construirFormulario() {
  const principales = $("#campos-principales");
  const indicadores = $("#campos-indicadores");
  let n = 0;
  estado.modelo.variables.forEach((v) => {
    const campo = crearCampo(v);
    if (PRINCIPALES[v.nombre]) principales.append(campo);
    else { indicadores.append(campo); n += 1; }
  });
  $("#num-indicadores").textContent = n;
}

function llenarSelector() {
  const sel = $("#selector-obra");
  estado.obras.forEach((o) => {
    const riesgo = o.riesgo_real ? "con sobrecosto > 50 %" : "sin sobrecosto > 50 %";
    sel.add(new Option(`${o.id_proyecto} · localidad ${o.localidad} · ${fmt(o.area_construida_m2, 0)} m² · ${riesgo}`, o.id_proyecto));
  });
  sel.addEventListener("change", () => cargarObra(sel.value));
}

async function cargarObra(id) {
  estado.obraActual = await llamar(`/api/obras-ejemplo/${encodeURIComponent(id)}`);
  const form = $("#formulario");
  Object.entries(estado.obraActual.variables).forEach(([k, x]) => {
    const c = form.elements[k];
    if (c) { c.value = x; c.parentElement.classList.remove("modificado"); }
  });
  $("#error-formulario").hidden = true;
}

function leerFormulario() {
  const variables = {};
  estado.modelo.variables.forEach((v) => {
    const texto = $("#formulario").elements[v.nombre].value;
    variables[v.nombre] = texto === "" ? null : Number(texto);
  });
  return variables;
}

function hayCambios() {
  return document.querySelectorAll("#formulario .campo.modificado").length > 0;
}

function pintarResultado(r) {
  const caja = $("#resultado");
  caja.classList.remove("vacio");
  caja.querySelector(".indicacion").hidden = true;
  caja.querySelector(".contenido").hidden = false;
  $(".semaforo").className = `semaforo ${r.nivel}`;
  $("#prob").textContent = fmt(r.probabilidad_pct, 1);
  const chip = $("#etiqueta-nivel");
  chip.className = `chip ${r.nivel}`;
  chip.textContent = `${NIVELES[r.nivel]}${r.alerta ? " · alerta" : ""}`;
  const ua = r.umbrales.amarillo * 100, ur = r.umbrales.rojo * 100;
  const barra = $(".barra-escala");
  barra.style.setProperty("--ua", `${ua}%`);
  barra.style.setProperty("--ur", `${ur}%`);
  $("#marcador").style.left = `${Math.min(100, r.probabilidad_pct)}%`;
  $("#marca-amarillo").style.left = `${ua}%`;
  $("#marca-amarillo").textContent = `${fmt(ua, 0)} %`;
  $("#marca-rojo").style.left = `${ur}%`;
  $("#marca-rojo").textContent = `${fmt(ur, 0)} %`;
  $("#mensaje").textContent = r.mensaje;
  const lista = $("#factores");
  lista.innerHTML = "";
  r.factores.forEach((f) => {
    const li = document.createElement("li");
    const sube = f.efecto > 0;
    li.innerHTML = `<span class="${sube ? "sube" : "baja"}">${sube ? "▲" : "▼"} ${f.sentido}</span> · ${f.descripcion}: <b>${fmt(f.valor, 4)}</b>`
      + ` ${f.unidad || ""} (típico ${fmt(f.valor_tipico, 4)}) · cambia la probabilidad ${fmt(Math.abs(f.efecto) * 100, 1)} puntos`;
    lista.append(li);
  });
  const adv = $("#advertencias");
  adv.hidden = r.advertencias.length === 0;
  adv.innerHTML = r.advertencias.map((a) => `⚠ ${a}`).join("<br>");
  const real = $("#real");
  if (estado.obraActual && !hayCambios()) {
    const o = estado.obraActual;
    real.hidden = false;
    real.innerHTML = `Dato real de <b>${o.id_proyecto}</b>: terminó con un sobrecosto de <b>${fmt(o.sobrecosto_real_pct, 1)} %</b>`
      + ` (${o.riesgo_real ? "sí superó" : "no superó"} el 50 %).`;
  } else {
    real.hidden = false;
    real.textContent = "Valores modificados a mano: no hay dato real para comparar.";
  }
  $("#tiempo").textContent = `${fmt(r.tiempo_ms, 1)} ms`;
  $("#version").textContent = r.version_modelo;
  $("#json").textContent = JSON.stringify(r, null, 2);
}

async function evaluar(evento) {
  evento.preventDefault();
  const boton = $("#boton-evaluar");
  const error = $("#error-formulario");
  boton.disabled = true;
  error.hidden = true;
  try {
    const cuerpo = { id_proyecto: estado.obraActual?.id_proyecto || "OBRA-MANUAL", variables: leerFormulario(), incluir_factores: true };
    pintarResultado(await llamar("/api/predecir", { method: "POST", body: JSON.stringify(cuerpo) }));
  } catch (e) {
    error.hidden = false;
    error.textContent = `${e.message}${e.errores.length ? ":\n" + e.errores.map((x) => `• ${x.variable}: ${x.problema}`).join("\n") : ""}`;
  } finally {
    boton.disabled = false;
  }
}

async function evaluarLote() {
  const boton = $("#boton-lote");
  boton.disabled = true;
  boton.textContent = "Evaluando…";
  try {
    const completas = await llamar("/api/obras-ejemplo?completo=true");
    const cuerpo = { obras: completas.map((o) => ({ id_proyecto: o.id_proyecto, variables: o.variables, incluir_factores: false })) };
    const r = await llamar("/api/predecir/lote", { method: "POST", body: JSON.stringify(cuerpo) });
    const real = Object.fromEntries(completas.map((o) => [o.id_proyecto, o]));
    const cruce = { verde: [0, 0], amarillo: [0, 0], rojo: [0, 0] };
    let vp = 0, fp = 0, fn = 0;
    const filas = r.resultados.map((x) => {
      const o = real[x.id_proyecto];
      const res = x.resultado;
      cruce[res.nivel][o.riesgo_real] += 1;
      if (res.alerta && o.riesgo_real) vp += 1;
      if (res.alerta && !o.riesgo_real) fp += 1;
      if (!res.alerta && o.riesgo_real) fn += 1;
      const lectura = res.alerta === Boolean(o.riesgo_real) ? "Coincide" : res.alerta ? "Falsa alarma" : "Riesgo no detectado";
      return `<tr><td>${o.id_proyecto}</td><td>${fmt(res.probabilidad_pct, 1)} %</td><td><span class="chip ${res.nivel}">${res.nivel}</span></td>`
        + `<td>${fmt(o.sobrecosto_real_pct, 1)} %</td><td>${o.riesgo_real ? "Sí" : "No"}</td><td>${lectura}</td></tr>`;
    });
    const tabla = $("#tabla-lote");
    tabla.querySelector("tbody").innerHTML = filas.join("");
    tabla.hidden = false;
    const recall = vp / (vp + fn), precision = vp / (vp + fp);
    const res = $("#resumen-lote");
    res.hidden = false;
    res.innerHTML = `
      <div class="kpi"><b>${r.evaluadas}</b><span>obras evaluadas en ${fmt(r.tiempo_ms, 0)} ms</span></div>
      <div class="kpi"><b>${vp} de ${vp + fn}</b><span>obras con sobrecosto real que recibieron alerta roja (recall ${fmt(recall, 3)})</span></div>
      <div class="kpi"><b>${vp} de ${vp + fp}</b><span>alertas rojas que fueron sobrecosto real (precisión ${fmt(precision, 3)})</span></div>
      <div class="kpi"><b>${cruce.verde[1]}</b><span>obras con sobrecosto real que quedaron en verde</span></div>
      <div style="grid-column: 1 / -1">
        <table class="cruce"><tr><th>Semáforo</th><th>Sin sobrecosto real</th><th>Con sobrecosto real</th></tr>
        ${["verde", "amarillo", "rojo"].map((k) => `<tr><td><span class="chip ${k}">${k}</span></td><td>${cruce[k][0]}</td><td>${cruce[k][1]}</td></tr>`).join("")}
        </table>
      </div>`;
  } catch (e) {
    alertaPanel("#resumen-lote", e);
  } finally {
    boton.disabled = false;
    boton.textContent = "Evaluar las 75 obras";
  }
}

function alertaPanel(selector, e) {
  const c = $(selector);
  c.hidden = false;
  c.innerHTML = `<p class="error">${e.message}</p>`;
}

function pintarModelo() {
  const m = estado.modelo;
  const mt = m.metricas_prueba;
  const filas = m.variables.map((v) => `<tr><td>${v.nombre}</td><td>${v.descripcion}</td><td>${v.unidad || "–"}</td>`
    + `<td>${fmt(v.minimo, 4)} – ${fmt(v.maximo, 4)}</td></tr>`).join("");
  $("#info-modelo").innerHTML = `
    <h2>${m.nombre} v${m.version}</h2>
    <p>${m.objetivo}. ${m.origen}.</p>
    <div class="info-grid">
      <div class="kpi"><b>${fmt(m.umbrales.rojo * 100, 0)} %</b><span>umbral rojo (alerta)</span></div>
      <div class="kpi"><b>${fmt(m.umbrales.amarillo * 100, 0)} %</b><span>umbral amarillo (seguimiento)</span></div>
      <div class="kpi"><b>${fmt(mt.recall, 3)}</b><span>recall en prueba: proporción de sobrecostos detectados</span></div>
      <div class="kpi"><b>${fmt(mt.precision, 3)}</b><span>precisión en prueba: alertas que sí eran sobrecosto</span></div>
      <div class="kpi"><b>${fmt(mt.roc_auc, 3)}</b><span>AUC en prueba (0.5 = azar, 1 = perfecto)</span></div>
      <div class="kpi"><b>${m.obras.entrenamiento} / ${m.obras.prueba}</b><span>obras de entrenamiento / prueba</span></div>
    </div>
    <h3>Datos y entrenamiento</h3>
    <p>${m.dataset}. Algoritmo: ${m.algoritmo}. Entrenado: ${m.fecha_entrenamiento}. Regla del umbral: ${m.regla_umbral}.</p>
    <h3>Variables que recibe la API (${m.variables.length})</h3>
    <div class="tabla-envoltura"><table><thead><tr><th>Variable</th><th>Descripción</th><th>Unidad</th><th>Rango de entrenamiento</th></tr></thead><tbody>${filas}</tbody></table></div>`;
}

function activarPestanas() {
  document.querySelectorAll(".pestana").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".pestana").forEach((x) => x.classList.toggle("activa", x === b));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("activo", p.id === b.dataset.panel));
  }));
}

async function iniciar() {
  activarPestanas();
  if (!(await revisarSalud())) return;
  estado.modelo = await llamar("/api/modelo");
  estado.obras = await llamar("/api/obras-ejemplo");
  construirFormulario();
  llenarSelector();
  pintarModelo();
  await cargarObra(estado.obras[0].id_proyecto);
  $("#formulario").addEventListener("submit", evaluar);
  $("#boton-restaurar").addEventListener("click", () => cargarObra($("#selector-obra").value));
  $("#boton-lote").addEventListener("click", evaluarLote);
  setInterval(revisarSalud, 30000);
}

iniciar();
