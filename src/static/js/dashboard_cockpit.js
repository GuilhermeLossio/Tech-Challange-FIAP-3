const state = {
  analysisResult: null,
  flightState: null,
  map: null,
  markers: [],
  liveMarker: null,
  routeLayer: null,
  availableFlights: [],
};

const $ = (id) => document.getElementById(id);
const apiBase = () => $("api-base").value.replace(/\/$/, "");
const MAX_SUGGESTIONS = 8;

const ICAO_IATA_MAP = {
  GLO: "G3",
  TAM: "JJ",
  LAM: "JJ",
  AZU: "AD",
  VOE: "V7",
  ONE: "2Z",
  MAP: "7M",
  DAL: "DL",
  UAL: "UA",
  AAL: "AA",
  BAW: "BA",
  AFR: "AF",
  KLM: "KL",
  IBE: "IB",
  LAN: "LA",
  TAP: "TP",
  DLH: "LH",
  RYR: "FR",
  EZY: "U2",
  SWR: "LX",
};

const AIRPORT_COORDS = {
  GRU: [-23.4356, -46.4731], CGH: [-23.6261, -46.6556], GIG: [-22.8099, -43.2505],
  SDU: [-22.9175, -43.1631], BSB: [-15.8711, -47.9186], CNF: [-19.6244, -43.9719],
  SSA: [-12.9108, -38.3225], FOR: [-3.7762, -38.5323], REC: [-8.1265, -34.9236],
  POA: [-29.9944, -51.1713], CWB: [-25.5285, -49.1758], FLN: [-27.6703, -48.5525],
  VCP: [-23.0074, -47.1344], NAT: [-5.9114, -35.2477], MCZ: [-9.5108, -35.7917],
  JPA: [-7.1455, -34.9503], THE: [-5.06, -42.8235], AJU: [-10.984, -37.0731],
  MAO: [-3.0386, -60.0498], BEL: [-1.3793, -48.4762], MCP: [4.0, -52.0],
  LDB: [-23.3336, -51.1302], IGU: [-25.5963, -54.4872], PVH: [-8.7093, -63.9023],
  MIA: [25.7959, -80.287], JFK: [40.6413, -73.7781], LAX: [33.9428, -118.4108],
  LHR: [51.4775, -0.4614], CDG: [49.0097, 2.5479], AMS: [52.3086, 4.7639],
  MAD: [40.4936, -3.5668], FCO: [41.8003, 12.2389], FRA: [50.0379, 8.5622],
  SCL: [-33.3928, -70.7856], EZE: [-34.8222, -58.5358], BOG: [4.7016, -74.1469],
  LIM: [-12.0219, -77.1143], MXP: [45.6306, 8.7231], MAN: [53.3619, -2.273],
};

function icaoToIata(icao) {
  return ICAO_IATA_MAP[(icao || "").toUpperCase()] || "";
}

function normalizeFlightToken(value) {
  return String(value || "").trim().toUpperCase();
}

function extractAvailableFlights(states) {
  const unique = new Map();
  for (const row of states || []) {
    const [icao24, callsign, country] = row;
    const token = normalizeFlightToken(callsign);
    if (!token) continue;
    if (!unique.has(token)) {
      unique.set(token, {
        callsign: token,
        country: country || "-",
        icao24: icao24 || "-",
      });
    }
  }
  state.availableFlights = Array.from(unique.values()).sort((a, b) => a.callsign.localeCompare(b.callsign));
}

function findSimilarFlights(query) {
  const token = normalizeFlightToken(query);
  const source = state.availableFlights || [];
  if (!source.length) return [];
  if (!token) return source.slice(0, MAX_SUGGESTIONS);

  const starts = [];
  const contains = [];
  for (const flight of source) {
    if (flight.callsign.startsWith(token)) {
      starts.push(flight);
    } else if (flight.callsign.includes(token)) {
      contains.push(flight);
    }
  }
  return starts.concat(contains).slice(0, MAX_SUGGESTIONS);
}

function hideSuggestions(containerId) {
  const container = $(containerId);
  if (!container) return;
  container.classList.remove("visible");
  container.innerHTML = "";
}

function renderSuggestions(containerId, matches, onPick) {
  const container = $(containerId);
  if (!container) return;

  if (!matches.length) {
    container.innerHTML = '<div class="suggestion-empty">Nenhum voo disponível no momento.</div>';
    container.classList.add("visible");
    return;
  }

  container.innerHTML = matches.map((flight) => `
    <button type="button" class="suggestion-item" data-callsign="${flight.callsign}">
      <span class="suggestion-callsign">${flight.callsign}</span>
      <span class="suggestion-meta">${flight.country}</span>
    </button>
  `).join("");
  container.classList.add("visible");

  container.querySelectorAll(".suggestion-item").forEach((btn) => {
    btn.addEventListener("mousedown", (event) => {
      event.preventDefault();
      onPick(btn.dataset.callsign || "");
    });
  });
}

function showFlightLookupSuggestions(query) {
  const matches = findSimilarFlights(query);
  renderSuggestions("flight-suggestions", matches, (callsign) => {
    $("flight-number-input").value = callsign;
    hideSuggestions("flight-suggestions");
  });
}

function showQuickSearchSuggestions(query) {
  const matches = findSimilarFlights(query);
  renderSuggestions("quick-search-suggestions", matches, (callsign) => {
    $("quick-search").value = callsign;
    hideSuggestions("quick-search-suggestions");
    quickSearch();
  });
}

function refreshSuggestionPanels() {
  const flightInput = $("flight-number-input");
  const quickInput = $("quick-search");
  if (document.activeElement === flightInput) {
    showFlightLookupSuggestions(flightInput.value);
  }
  if (document.activeElement === quickInput) {
    showQuickSearchSuggestions(quickInput.value);
  }
}

function bindAutocomplete() {
  const flightInput = $("flight-number-input");
  const quickInput = $("quick-search");
  if (!flightInput || !quickInput) return;

  flightInput.addEventListener("focus", () => showFlightLookupSuggestions(flightInput.value));
  flightInput.addEventListener("input", () => showFlightLookupSuggestions(flightInput.value));
  flightInput.addEventListener("blur", () => {
    setTimeout(() => hideSuggestions("flight-suggestions"), 120);
  });

  quickInput.addEventListener("focus", () => showQuickSearchSuggestions(quickInput.value));
  quickInput.addEventListener("input", () => showQuickSearchSuggestions(quickInput.value));
  quickInput.addEventListener("blur", () => {
    setTimeout(() => hideSuggestions("quick-search-suggestions"), 120);
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest(".suggestions-host")) {
      hideSuggestions("flight-suggestions");
      hideSuggestions("quick-search-suggestions");
    }
  });
}

async function checkApiHealth() {
  const chip = $("api-health-chip");
  const pill = $("api-status-pill");
  chip.className = "status-chip";
  chip.innerHTML = '<span class="spinner-inline"></span> conectando...';
  try {
    const r = await fetch(apiBase() + "/health", { signal: AbortSignal.timeout(4000) });
    if (!r.ok) {
      throw new Error("health");
    }
    chip.className = "status-chip ok";
    chip.textContent = "conectado";
    pill.style.borderColor = "rgba(127,255,110,.5)";
    pill.style.color = "var(--accent3)";
  } catch {
    chip.className = "status-chip err";
    chip.textContent = "offline";
    pill.style.borderColor = "rgba(255,69,96,.4)";
    pill.style.color = "var(--danger)";
  }
}

function initMap() {
  state.map = L.map("live-map", { zoomControl: true, attributionControl: false }).setView([0, 0], 2);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", { maxZoom: 18 }).addTo(state.map);
  fetchLivePlanes();
  setInterval(fetchLivePlanes, 15000);
}

async function fetchLivePlanes() {
  const url = "https://opensky-network.org/api/states/all?lamin=-55&lamax=15&lomin=-90&lomax=-30";
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (!r.ok) return;
    const data = await r.json();
    renderLivePlanes(data.states || []);
  } catch (e) {
    console.warn("Falha na busca do OpenSky:", e.message);
  }
}

function renderLivePlanes(states) {
  extractAvailableFlights(states);
  refreshSuggestionPanels();

  state.markers.forEach((m) => m.remove());
  state.markers = [];

  states.slice(0, 120).forEach((s) => {
    const [icao24, callsign, country, , , lon, lat, , , , vel, , , heading] = s;
    if (!lat || !lon) return;

    const icon = L.divIcon({
      className: "",
      html: `<div style="color:#00d4ff;font-size:14px;text-shadow:0 0 8px rgba(0,212,255,.7);transform:rotate(${heading || 0}deg)">✈</div>`,
      iconSize: [18, 18],
      iconAnchor: [9, 9],
    });

    const marker = L.marker([lat, lon], { icon })
      .bindPopup(
        `<div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#dce4f0;background:#111520;padding:8px;border-radius:6px">
          <b style="color:#00d4ff">${(callsign || "").trim() || icao24}</b><br>
          País: ${country}<br>
          Alt: ${s[7] != null ? Math.round(s[7]) + "m" : "-"}<br>
          Vel: ${vel != null ? Math.round(vel * 3.6) + "km/h" : "-"}
        </div>`,
        { className: "flight-popup" }
      )
      .addTo(state.map);
    state.markers.push(marker);
  });
}

async function lookupFlight() {
  const raw = $("flight-number-input").value.trim().toUpperCase();
  if (!raw) return;
  hideSuggestions("flight-suggestions");

  const btn = $("btn-lookup");
  const status = $("lookup-status");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-inline"></span>';
  status.textContent = "Consultando OpenSky...";

  try {
    const directUrl = `https://opensky-network.org/api/states/all?callsign=${encodeURIComponent(raw.padEnd(8))}`;
    let r = await fetch(directUrl, { signal: AbortSignal.timeout(8000) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    let data = await r.json();
    let states = data.states || [];

    if (!states.length) {
      r = await fetch(`https://opensky-network.org/api/states/all?callsign=${encodeURIComponent(raw)}`, {
        signal: AbortSignal.timeout(8000),
      });
      data = await r.json();
      states = data.states || [];
    }

    if (!states.length) {
      status.textContent = `Voo "${raw}" não encontrado no momento. Preenchimento manual disponível.`;
      autoFillFromIcao(raw);
      return;
    }

    fillFlightCard(states[0], raw);
    status.textContent = "";
  } catch (e) {
    status.textContent = `Erro: ${e.message}. Tente novamente.`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Buscar";
  }
}

function fillFlightCard(s, fallbackCallsign) {
  const [, cs, country, , , lon, lat, baroAlt, , , vel] = s;
  const realCallsign = (cs || "").trim() || fallbackCallsign;
  const airlineIata = icaoToIata(realCallsign.replace(/[0-9]/g, "").trim());

  $("fc-callsign").textContent = realCallsign;
  $("fc-country").textContent = country || "-";
  $("fc-alt").textContent = baroAlt != null ? `${Math.round(baroAlt)}m` : "-";
  $("fc-vel").textContent = vel != null ? `${Math.round(vel * 3.6)} km/h` : "-";
  $("fc-origin").textContent = "???";
  $("fc-dest").textContent = "???";

  if (airlineIata) $("f-airline").value = airlineIata;
  $("flight-info-card").classList.add("visible");

  if (lat && lon && state.map) {
    state.map.flyTo([lat, lon], 6, { duration: 1.5 });
    if (state.liveMarker) state.liveMarker.remove();
    const icon = L.divIcon({
      className: "",
      html: '<div style="color:#ff6b35;font-size:20px;text-shadow:0 0 16px rgba(255,107,53,.9)">✈</div>',
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });
    state.liveMarker = L.marker([lat, lon], { icon })
      .bindPopup(`<b style="color:#ff6b35">${realCallsign}</b><br>${country || ""}`)
      .addTo(state.map)
      .openPopup();
  }
}

function autoFillFromIcao(callsign) {
  const airlineIata = icaoToIata(callsign.replace(/[0-9]/g, "").trim());
  if (airlineIata) $("f-airline").value = airlineIata;
}

function drawRoute(originCode, destCode) {
  if (!state.map) return;
  if (state.routeLayer) {
    state.routeLayer.remove();
    state.routeLayer = null;
  }
  const o = AIRPORT_COORDS[(originCode || "").toUpperCase()];
  const d = AIRPORT_COORDS[(destCode || "").toUpperCase()];
  if (!o || !d) return;

  const pts = [];
  for (let t = 0; t <= 1; t += 0.02) {
    const lat = o[0] + (d[0] - o[0]) * t + Math.sin(Math.PI * t) * 0.9;
    const lon = o[1] + (d[1] - o[1]) * t;
    pts.push([lat, lon]);
  }

  state.routeLayer = L.polyline(pts, {
    color: "#00d4ff",
    weight: 2,
    opacity: 0.7,
    dashArray: "6 4",
  }).addTo(state.map);

  L.circleMarker(o, { radius: 6, fillColor: "#ff6b35", fillOpacity: 1, color: "transparent" }).addTo(state.map);
  L.circleMarker(d, { radius: 6, fillColor: "#00d4ff", fillOpacity: 1, color: "transparent" }).addTo(state.map);

  state.map.fitBounds(L.latLngBounds([o, d]).pad(0.2), { animate: true, duration: 1 });
}

function buildHourlyData(airline, origin, dest) {
  const base = [0.18, 0.14, 0.12, 0.11, 0.12, 0.15, 0.22, 0.31, 0.38, 0.35, 0.33, 0.3, 0.35, 0.4, 0.43, 0.42, 0.41, 0.46, 0.49, 0.51, 0.48, 0.42, 0.35, 0.25];
  const seed = (airline + origin + dest).split("").reduce((a, c) => a + c.charCodeAt(0), 0);
  const rng = (i) => ((seed * i * 7 + i * 13) % 100) / 700;
  return base.map((v, i) => Math.min(0.95, Math.max(0.05, v + rng(i) - 0.07)));
}

let hourlyChart = null;
function renderHourlyChart(airline, origin, dest, currentHour) {
  const canvas = $("chart-hourly");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const data = buildHourlyData(airline || "XX", origin || "XXX", dest || "XXX");
  const labels = Array.from({ length: 24 }, (_, i) => `${String(i).padStart(2, "0")}h`);
  const highlight = data.map((_, i) => (i === currentHour ? "rgba(255,107,53,.9)" : "rgba(0,212,255,.6)"));

  if (hourlyChart) hourlyChart.destroy();
  hourlyChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: data.map((v) => +(v * 100).toFixed(1)),
        backgroundColor: highlight,
        borderRadius: 3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: "rgba(255,255,255,.04)" }, ticks: { color: "#4a5568", font: { size: 10 } } },
        y: {
          grid: { color: "rgba(255,255,255,.04)" },
          ticks: { color: "#4a5568", font: { size: 10 }, callback: (v) => `${v}%` },
          min: 0,
          max: 80,
        },
      },
    },
  });
}

const CARRIERS = ["G3", "JJ", "AD", "LA", "AA", "DL", "UA", "KL", "AF", "IB"];
const CARRIER_NAMES = {
  G3: "Gol", JJ: "LATAM", AD: "Azul", LA: "LATAM", AA: "American",
  DL: "Delta", UA: "United", KL: "KLM", AF: "Air France", IB: "Iberia",
};

function buildCarrierData(origin, dest, myAirline) {
  const seed = (origin + dest).split("").reduce((a, c) => a + c.charCodeAt(0), 0);
  const carriers = CARRIERS.filter((c) => c !== myAirline).slice(0, 5);
  if (myAirline) carriers.unshift(myAirline);
  return carriers.map((c, i) => {
    const base = 0.25 + ((seed * (i + 1) * 3) % 100) / 250;
    return { airline: c, name: CARRIER_NAMES[c] || c, prob: Math.min(0.85, Math.max(0.1, base)) };
  });
}

let carrierChart = null;
function renderCarrierChart(origin, dest, myAirline) {
  const wrap = $("carrier-wrap");
  if (!wrap) return;
  wrap.innerHTML = '<canvas id="chart-carrier"></canvas>';
  const ctx = $("chart-carrier").getContext("2d");
  const data = buildCarrierData(origin, dest, myAirline);
  const colors = data.map((d) => (d.airline === myAirline ? "#ff6b35" : "rgba(127,255,110,.7)"));

  if (carrierChart) carrierChart.destroy();
  carrierChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: data.map((d) => d.name),
      datasets: [{ data: data.map((d) => +(d.prob * 100).toFixed(1)), backgroundColor: colors, borderRadius: 4 }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,.04)" },
          ticks: { color: "#4a5568", font: { size: 10 }, callback: (v) => `${v}%` },
          max: 80,
        },
        y: { grid: { display: false }, ticks: { color: "#dce4f0", font: { size: 11 } } },
      },
    },
  });
}

function renderFactors(topFactors) {
  const wrap = $("radar-wrap");
  if (!wrap) return;
  if (!topFactors || !topFactors.length) {
    wrap.innerHTML = '<div class="empty-state"><div class="empty-icon">📡</div><div class="empty-text">Aguardando análise</div></div>';
    return;
  }

  const parsed = topFactors.map((f) => ({ name: f.feature.replace(/_/g, " "), val: parseFloat((f.impact || "").replace("%", "")) || 0 }));
  const maxAbs = Math.max(...parsed.map((p) => Math.abs(p.val)), 1);
  wrap.innerHTML = `<div class="factors-list">${parsed.map((p) => {
    const isPos = p.val > 0;
    const barW = Math.min(100, (Math.abs(p.val) / maxAbs) * 100);
    return `
      <div class="factor-row">
        <div class="factor-name-text">${p.name}</div>
        <div class="factor-bar-wrap">
          <div class="factor-bar" style="width:${barW}%;background:${isPos ? "var(--danger)" : "var(--accent3)"}"></div>
        </div>
        <div class="factor-val ${isPos ? "pos" : "neg"}">${p.val > 0 ? "+" : ""}${p.val.toFixed(1)}%</div>
      </div>`;
  }).join("")}</div>`;
}

function showResult(data) {
  const prob = Number(data.delay_probability || 0);
  const pct = Math.round(prob * 100);
  const level = String(data.risk_level || "LOW").toUpperCase();
  const bar = $("result-bar");
  if (bar) bar.classList.add("visible");

  const badge = $("risk-badge");
  if (badge) {
    badge.className = `risk-badge risk-${level}`;
    badge.textContent = { LOW: "BAIXO RISCO", MEDIUM: "RISCO MÉDIO", HIGH: "ALTO RISCO" }[level] || level;
  }
  $("prob-num").textContent = String(pct);
  $("advice-inline").textContent = data.advice || "";
  $("prob-num").style.color = level === "HIGH" ? "var(--danger)" : level === "MEDIUM" ? "var(--warn)" : "var(--accent3)";
}

async function runAnalysis() {
  const origin = $("f-origin").value.trim().toUpperCase();
  const dest = $("f-dest").value.trim().toUpperCase();
  const airline = $("f-airline").value.trim().toUpperCase();
  const dep = $("f-dep").value.trim();
  const date = $("f-date").value;
  const dist = $("f-dist").value;

  if (!origin || !dest || !airline || !dep) {
    alert("Preencha Origem, Destino, Companhia e Horário de Partida.");
    return;
  }

  const btn = $("btn-analyze");
  btn.disabled = true;
  btn.textContent = "Analisando...";

  drawRoute(origin, dest);
  const hour = Math.floor(parseInt(dep, 10) / 100);
  renderHourlyChart(airline, origin, dest, Number.isNaN(hour) ? -1 : hour);
  renderCarrierChart(origin, dest, airline);

  const payload = {
    origin_airport: origin,
    destination_airport: dest,
    airline,
    scheduled_departure: parseInt(dep, 10),
  };
  if (date) payload.flight_date = date;
  if (dist) payload.distance = parseFloat(dist);

  try {
    const r = await fetch(apiBase() + "/advise", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(10000),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    state.analysisResult = data;
    showResult(data);
    renderFactors(data.top_factors);
  } catch (e) {
    const mockProb = 0.34 + Math.random() * 0.3;
    const mockData = {
      delay_probability: mockProb,
      risk_level: mockProb >= 0.7 ? "HIGH" : mockProb >= 0.4 ? "MEDIUM" : "LOW",
      top_factors: [
        { feature: "CARRIER_DELAY_RATE", impact: "+8.2%" },
        { feature: "ORIGIN_DELAY_RATE", impact: "+5.1%" },
        { feature: "ROUTE_DELAY_RATE", impact: "-2.3%" },
      ],
      advice: `[Modo de demonstração - API offline] Rota ${origin}->${dest} tem uma probabilidade simulada de ${(mockProb * 100).toFixed(0)}%.`,
    };
    showResult(mockData);
    renderFactors(mockData.top_factors);
  } finally {
    btn.disabled = false;
    btn.textContent = "ANALISAR RISCO DE ATRASO";
  }
}

function quickSearch() {
  const val = $("quick-search").value.trim().toUpperCase();
  if (!val) return;
  hideSuggestions("quick-search-suggestions");
  if (/^[A-Z]{2,3}[0-9]{1,4}$/.test(val)) {
    $("flight-number-input").value = val;
    lookupFlight();
    return;
  }
  if (/^[A-Z]{3}$/.test(val)) {
    $("f-origin").value = val;
  }
}

async function loadWeeklyPredictions() {
  const weeklyPredictionsWrap = document.getElementById('weekly-predictions-wrap');
  const table = document.getElementById('weekly-predictions-table');
  const emptyState = weeklyPredictionsWrap.querySelector('.empty-state');
  
  try {
    const response = await fetch(`${apiBase()}/api/upcoming_flights?limit=60`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();

    if (data.predictions && data.predictions.length > 0) {
      populateTable(table, data.predictions);
      table.classList.remove('hidden');
      emptyState.classList.add('hidden');
    } else {
      const detail = data.detail || 'Nenhum voo futuro encontrado.';
      emptyState.querySelector('.empty-text').textContent = detail;
    }
  } catch (error) {
    console.error('Falha ao carregar lista de voos:', error);
    if(emptyState.querySelector('.empty-text')) {
      emptyState.querySelector('.empty-text').textContent = 'Falha ao carregar lista de voos.';
    }
  }
}

function populateTable(table, predictions) {
  const thead = table.querySelector('thead');
  const tbody = table.querySelector('tbody');
  thead.innerHTML = '';
  tbody.innerHTML = '';

  if (predictions.length === 0) return;

  const headers = Object.keys(predictions[0]);
  const headerRow = document.createElement('tr');
  headers.forEach(headerText => {
    const th = document.createElement('th');
    th.textContent = headerText;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  predictions.forEach(prediction => {
    const row = document.createElement('tr');
    headers.forEach(header => {
      const cell = document.createElement('td');
      cell.textContent = prediction[header];
      row.appendChild(cell);
    });
    tbody.appendChild(row);
  });
}

window.addEventListener("DOMContentLoaded", () => {
  $("f-date").value = new Date().toISOString().slice(0, 10);
  bindAutocomplete();
  initMap();
  checkApiHealth();
  loadWeeklyPredictions();
  renderHourlyChart("XX", "XXX", "XXX", -1);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && document.activeElement === $("flight-number-input")) {
      lookupFlight();
    }
  });
});
