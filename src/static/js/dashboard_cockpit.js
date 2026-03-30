const state = {
  analysisResult: null,
  flightState: null,
  map: null,
  markers: [],
  liveMarker: null,
  routeLayer: null,
  availableFlights: [],
  lastLiveFlights: [],
  upcomingFlights: [],
  liveFeedStatus: null,
};

const $ = (id) => document.getElementById(id);
const apiBaseInput = () => $("api-base");

function normalizeApiBase(rawValue) {
  const raw = String(rawValue || "").trim();
  if (!raw || raw === "/") return "";

  const candidate = /^[a-z][a-z0-9+.-]*:\/\//i.test(raw)
    ? raw
    : raw.startsWith("/")
      ? raw
      : `/${raw.replace(/^\/+/, "")}`;

  try {
    const resolved = new URL(candidate, window.location.origin);
    if (resolved.origin === window.location.origin) {
      return resolved.pathname === "/" ? "" : resolved.pathname.replace(/\/$/, "");
    }
    return resolved.toString().replace(/\/$/, "");
  } catch {
    return "";
  }
}

const apiBase = () => normalizeApiBase(apiBaseInput()?.value);
const MAX_SUGGESTIONS = 8;
const LIVE_FEED_REGION = "brazil";
const LIVE_FEED_REFRESH_MS = 60_000;
const LIVE_FEED_FETCH_TIMEOUT_MS = 15_000;

// ── Chart.js defaults aligned to the dark theme ──────────────
if (window.Chart?.defaults) {
  Chart.defaults.color = "#6b7280";
  Chart.defaults.borderColor = "rgba(255,255,255,.06)";
  Chart.defaults.font.family = "'DM Sans', sans-serif";
}

const ICAO_IATA_MAP = {
  GLO: "G3", TAM: "JJ", LAM: "JJ", AZU: "AD", VOE: "V7",
  ONE: "2Z", MAP: "7M", DAL: "DL", UAL: "UA", AAL: "AA",
  BAW: "BA", AFR: "AF", KLM: "KL", IBE: "IB", LAN: "LA",
  TAP: "TP", DLH: "LH", RYR: "FR", EZY: "U2", SWR: "LX",
};

const AIRPORT_COORDS = {
  GRU: [-23.4356, -46.4731], CGH: [-23.6261, -46.6556], GIG: [-22.8099, -43.2505],
  SDU: [-22.9175, -43.1631], BSB: [-15.8711, -47.9186], CNF: [-19.6244, -43.9719],
  SSA: [-12.9108, -38.3225], FOR: [-3.7762, -38.5323], REC: [-8.1265, -34.9236],
  POA: [-29.9944, -51.1713], CWB: [-25.5285, -49.1758], FLN: [-27.6703, -48.5525],
  VCP: [-23.0074, -47.1344], NAT: [-5.9114, -35.2477], MCZ: [-9.5108, -35.7917],
  JPA: [-7.1455, -34.9503], THE: [-5.06, -42.8235],   AJU: [-10.984, -37.0731],
  MAO: [-3.0386, -60.0498], BEL: [-1.3793, -48.4762], MCP: [4.0, -52.0],
  LDB: [-23.3336, -51.1302], IGU: [-25.5963, -54.4872], PVH: [-8.7093, -63.9023],
  MIA: [25.7959, -80.287],  JFK: [40.6413, -73.7781], LAX: [33.9428, -118.4108],
  LHR: [51.4775, -0.4614],  CDG: [49.0097, 2.5479],   AMS: [52.3086, 4.7639],
  MAD: [40.4936, -3.5668],  FCO: [41.8003, 12.2389],  FRA: [50.0379, 8.5622],
  SCL: [-33.3928, -70.7856], EZE: [-34.8222, -58.5358], BOG: [4.7016, -74.1469],
  LIM: [-12.0219, -77.1143], MXP: [45.6306, 8.7231],  MAN: [53.3619, -2.273],
};

// ── Helpers ──────────────────────────────────────────────────
function icaoToIata(icao) {
  return ICAO_IATA_MAP[(icao || "").toUpperCase()] || "";
}
function normalizeFlightToken(value) {
  return String(value || "").trim().toUpperCase();
}
function hasValidCoordinates(latitude, longitude) {
  return Number.isFinite(latitude) && Number.isFinite(longitude);
}
function buildScheduledCallsign(flight) {
  return normalizeFlightToken(`${flight?.airline || ""}${flight?.flight_number || ""}`);
}

// ── Autocomplete / suggestions ───────────────────────────────
function findSimilarFlights(query) {
  const token = normalizeFlightToken(query);
  const source = state.availableFlights || [];
  if (!source.length) return [];
  if (!token) return source.slice(0, MAX_SUGGESTIONS);
  const starts = [], contains = [];
  for (const flight of source) {
    if (flight.callsign.startsWith(token)) starts.push(flight);
    else if (flight.callsign.includes(token)) contains.push(flight);
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
    container.innerHTML = '<div class="suggestion-empty">No flights available at the moment.</div>';
    container.classList.add("visible");
    return;
  }
  container.innerHTML = matches.map((f) => `
    <button type="button" class="suggestion-item" data-callsign="${f.callsign}">
      <span class="suggestion-callsign">${f.callsign}</span>
      <span class="suggestion-meta">${f.country}</span>
    </button>`).join("");
  container.classList.add("visible");
  container.querySelectorAll(".suggestion-item").forEach((btn) => {
    btn.addEventListener("mousedown", (e) => {
      e.preventDefault();
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

function refreshSuggestionPanels() {
  const flightInput = $("flight-number-input");
  if (flightInput && document.activeElement === flightInput) {
    showFlightLookupSuggestions(flightInput.value);
  }
}

function bindAutocomplete() {
  const flightInput = $("flight-number-input");
  if (!flightInput) return;
  flightInput.addEventListener("focus", () => showFlightLookupSuggestions(flightInput.value));
  flightInput.addEventListener("input",  () => showFlightLookupSuggestions(flightInput.value));
  flightInput.addEventListener("blur",   () => setTimeout(() => hideSuggestions("flight-suggestions"), 120));
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".suggestions-host")) hideSuggestions("flight-suggestions");
  });
}

// ── API health ───────────────────────────────────────────────
async function checkApiHealth() {
  const chip = $("api-health-chip");
  if (!chip) return;
  chip.className = "status-chip";
  chip.textContent = "connecting...";
  try {
    const r = await fetch(apiBase() + "/health", { signal: AbortSignal.timeout(4000) });
    if (!r.ok) throw new Error("health");
    chip.className = "status-chip connected";
    chip.textContent = "connected";
  } catch {
    chip.className = "status-chip";
    chip.textContent = "offline";
  }
}

function bindApiBaseField() {
  const input = apiBaseInput();
  if (!input) return;
  input.addEventListener("change", () => {
    input.value = apiBase();
    checkApiHealth();
  });
  input.addEventListener("blur", () => {
    input.value = apiBase();
    checkApiHealth();
  });
  input.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    input.value = apiBase();
    input.blur();
  });
}

// ── Map ──────────────────────────────────────────────────────
function initMap() {
  const container = $("live-map");
  if (!window.L) {
    console.warn("Leaflet is unavailable; skipping live map initialization.");
    return;
  }
  if (!container || state.map || container._leaflet_id) return;
  state.map = L.map("live-map", { zoomControl: true, attributionControl: false }).setView([-15, -50], 4);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", { maxZoom: 18 }).addTo(state.map);
  fetchLivePlanes();
  setInterval(fetchLivePlanes, LIVE_FEED_REFRESH_MS);
}

async function fetchLivePlanes() {
  const url = `${apiBase()}/api/live_flights?region=${LIVE_FEED_REGION}&limit=180&degraded=1`;
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(LIVE_FEED_FETCH_TIMEOUT_MS) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    state.lastLiveFlights = data.flights || [];
    state.liveFeedStatus = {
      available: data.live_available !== false,
      detail: data.detail || "",
      providerStatus: data.provider_status || "ok",
      staleCache: data.stale_cache === true,
      cacheAgeSec: Number.isFinite(data.cache_age_sec) ? data.cache_age_sec : null,
    };
    updateAvailableFlightsCatalog();
    renderLivePlanes(state.lastLiveFlights);
    if (data.live_available === false || data.stale_cache === true) {
      console.warn("Live feed unavailable:", data.detail || "OpenSky degraded mode");
    }
  } catch (e) {
    state.liveFeedStatus = {
      available: false,
      detail: e.message || "Unable to fetch live flights.",
      providerStatus: "network_error",
      staleCache: false,
      cacheAgeSec: null,
    };
    updateAvailableFlightsCatalog();
    console.warn("Failed to fetch live flights:", e.message);
  }
}

function renderLivePlanes(flights) {
  state.markers.forEach((m) => m.remove());
  state.markers = [];

  flights.forEach((flight) => {
    const {
      icao24,
      callsign,
      origin_country,
      longitude,
      latitude,
      altitude_ft,
      speed_kmh,
      on_ground,
      heading,
    } = flight;

    if (!hasValidCoordinates(latitude, longitude) || on_ground) return;

    const icon = L.divIcon({
      className: "",
      html: `<div style="color:#4f8ef7;font-size:13px;transform:rotate(${heading || 0}deg)">✈</div>`,
      iconSize: [16, 16],
      iconAnchor: [8, 8],
    });

    const marker = L.marker([latitude, longitude], { icon })
      .bindPopup(
        `<div style="font-family:'Space Mono',monospace;font-size:11px;background:#111520;color:#e8eaf0;padding:8px;border-radius:6px">
          <b style="color:#4f8ef7">${(callsign || "").trim() || icao24}</b><br>
          Country: ${origin_country || "-"}<br>
          Altitude: ${altitude_ft != null ? Math.round(altitude_ft) + " ft" : "-"}<br>
          Speed: ${speed_kmh != null ? Math.round(speed_kmh) + " km/h" : "-"}
        </div>`,
        { className: "flight-popup" }
      )
      .addTo(state.map);
    state.markers.push(marker);
  });
}

// Extracts available flights from live data
function collectAvailableFlightsFromObjects(flights) {
  const unique = new Map();
  for (const flight of flights || []) {
    const { callsign, origin_country, country, icao24 } = flight;
    const token = normalizeFlightToken(callsign);
    if (!token) continue;
    if (!unique.has(token)) {
      unique.set(token, {
        callsign: token,
        country: origin_country || country || "-",
        icao24: icao24 || "-",
        source: "live",
      });
    }
  }
  return Array.from(unique.values());
}

function collectAvailableFlightsFromSchedule(predictions) {
  const unique = new Map();
  for (const flight of predictions || []) {
    const callsign = buildScheduledCallsign(flight);
    if (!callsign) continue;
    if (!unique.has(callsign)) {
      unique.set(callsign, {
        callsign,
        country: flight.route || "Scheduled flight",
        icao24: "",
        source: "schedule",
        flight_date: flight.flight_date || "",
        scheduled_departure: flight.scheduled_departure || "",
        origin_airport: flight.origin_airport || "",
        destination_airport: flight.destination_airport || "",
        airline: flight.airline || "",
        flight_number: flight.flight_number || "",
      });
    }
  }
  return Array.from(unique.values());
}

function updateAvailableFlightsCatalog() {
  const merged = new Map();
  collectAvailableFlightsFromSchedule(state.upcomingFlights).forEach((flight) => {
    merged.set(flight.callsign, flight);
  });
  collectAvailableFlightsFromObjects(state.lastLiveFlights).forEach((flight) => {
    merged.set(flight.callsign, { ...(merged.get(flight.callsign) || {}), ...flight });
  });
  state.availableFlights = Array.from(merged.values()).sort((a, b) => a.callsign.localeCompare(b.callsign));
  refreshSuggestionPanels();
}

// ── Flight lookup ────────────────────────────────────────────
async function lookupFlight() {
  const rawCallsign = $("flight-number-input").value.trim().toUpperCase();
  if (!rawCallsign) return;
  hideSuggestions("flight-suggestions");

  const btn = $("btn-lookup");
  const status = $("lookup-status");
  btn.disabled = true;
  btn.textContent = "...";
  status.textContent = "Querying backend...";

  // 1. Find the ICAO24 code for the given callsign
  const flight = state.availableFlights.find(f => f.callsign === rawCallsign);
  if (!flight) {
    status.textContent = `Flight "${rawCallsign}" not found in live or scheduled data.`;
    autoFillFromIcao(rawCallsign);
    btn.disabled = false;
    btn.textContent = "Search";
    return;
  }
  if (!flight.icao24) {
    fillScheduledFlightCard(flight, rawCallsign);
    status.textContent = state.liveFeedStatus?.available === false
      ? "Live tracking is unavailable. Showing scheduled flight details."
      : "Showing scheduled flight details.";
    btn.disabled = false;
    btn.textContent = "Search";
    return;
  }

  // 2. Use the ICAO24 code to fetch detailed flight data
  try {
    const url = `${apiBase()}/api/live_flights/${flight.icao24}`;
    const r = await fetch(url, { signal: AbortSignal.timeout(LIVE_FEED_FETCH_TIMEOUT_MS) });
    const data = await r.json();

    if (!r.ok) {
      throw new Error(data.detail || `HTTP ${r.status}`);
    }

    // Preserve scheduled route metadata when live detail omits it.
    fillFlightCard({ ...(flight || {}), ...(data.flight || {}) }, rawCallsign);
    status.textContent = "";
  } catch (e) {
    if (flight.origin_airport || flight.destination_airport) {
      fillScheduledFlightCard(flight, rawCallsign);
      status.textContent = "Live tracking is unavailable. Showing scheduled flight details.";
    } else {
      status.textContent = `Error: ${e.message}`;
      autoFillFromIcao(rawCallsign);
    }
  } finally {
    btn.disabled = false;
    btn.textContent = "Search";
  }
}

function fillFlightCard(flight, fallbackCallsign) {
  if (!flight) {
    // If no flight data is returned, clear the card
    $("fc-callsign").textContent = fallbackCallsign || "-";
    $("fc-country").textContent  = "-";
    $("fc-alt").textContent      = "-";
    $("fc-vel").textContent      = "-";
    $("fc-origin").textContent   = "?";
    $("fc-dest").textContent     = "?";
    autoFillFromIcao(fallbackCallsign);
    return;
  }

  const {
    callsign,
    origin_country,
    country,
    altitude_ft,
    speed_kmh,
    latitude,
    longitude,
  } = flight;
  const realCallsign = (callsign || "").trim() || fallbackCallsign;
  const airlineIata = icaoToIata(realCallsign.replace(/[0-9]/g, "").trim());

  $("fc-callsign").textContent = realCallsign;
  $("fc-country").textContent  = origin_country || country || "-";
  $("fc-alt").textContent      = altitude_ft != null ? `${Math.round(altitude_ft)} ft` : "-";
  $("fc-vel").textContent      = speed_kmh != null ? `${Math.round(speed_kmh)} km/h` : "-";
  $("fc-origin").textContent   = flight.origin_airport || "?";
  $("fc-dest").textContent     = flight.destination_airport || "?";

  if (airlineIata) $("f-airline").value = airlineIata;
  if (flight.origin_airport) $("f-origin").value = flight.origin_airport;
  if (flight.destination_airport) $("f-dest").value = flight.destination_airport;

  if (hasValidCoordinates(latitude, longitude) && state.map) {
    state.map.flyTo([latitude, longitude], 8, { duration: 1.5 });
    if (state.liveMarker) state.liveMarker.remove();
    const icon = L.divIcon({
      className: "",
      html: '<div style="color:#f7c94f;font-size:20px;text-shadow:0 0 12px rgba(247,201,79,.8)">✈</div>',
      iconSize: [24, 24], iconAnchor: [12, 12],
    });
    state.liveMarker = L.marker([latitude, longitude], { icon })
      .bindPopup(`<b style="color:#f7c94f">${realCallsign}</b><br>${origin_country || country || ""}`)
      .addTo(state.map).openPopup();
  }
}

function fillScheduledFlightCard(flight, fallbackCallsign) {
  const realCallsign = flight?.callsign || fallbackCallsign || "-";
  const airlineIata = icaoToIata(realCallsign.replace(/[0-9]/g, "").trim());

  $("fc-callsign").textContent = realCallsign;
  $("fc-country").textContent = flight?.flight_date ? `Scheduled ${flight.flight_date}` : "Scheduled flight";
  $("fc-alt").textContent = "-";
  $("fc-vel").textContent = flight?.scheduled_departure ? `STD ${flight.scheduled_departure}` : "-";
  $("fc-origin").textContent = flight?.origin_airport || "?";
  $("fc-dest").textContent = flight?.destination_airport || "?";

  if (airlineIata) $("f-airline").value = airlineIata;
  if (flight?.origin_airport) $("f-origin").value = flight.origin_airport;
  if (flight?.destination_airport) $("f-dest").value = flight.destination_airport;
}

function autoFillFromIcao(callsign) {
  const airlineIata = icaoToIata(callsign.replace(/[0-9]/g, "").trim());
  if (airlineIata) $("f-airline").value = airlineIata;
}

// ── Route drawing ────────────────────────────────────────────
function drawRoute(originCode, destCode) {
  if (!state.map) return;
  if (state.routeLayer) { state.routeLayer.remove(); state.routeLayer = null; }
  const o = AIRPORT_COORDS[(originCode || "").toUpperCase()];
  const d = AIRPORT_COORDS[(destCode || "").toUpperCase()];
  if (!o || !d) return;
  const pts = [];
  for (let t = 0; t <= 1; t += 0.02) {
    pts.push([o[0] + (d[0] - o[0]) * t + Math.sin(Math.PI * t) * 0.9, o[1] + (d[1] - o[1]) * t]);
  }
  state.routeLayer = L.polyline(pts, { color: "#4f8ef7", weight: 2, opacity: 0.7, dashArray: "6 4" }).addTo(state.map);
  L.circleMarker(o, { radius: 6, fillColor: "#f7c94f", fillOpacity: 1, color: "transparent" }).addTo(state.map);
  L.circleMarker(d, { radius: 6, fillColor: "#4ff7a0", fillOpacity: 1, color: "transparent" }).addTo(state.map);
  state.map.fitBounds(L.latLngBounds([o, d]).pad(0.2), { animate: true, duration: 1 });
}

// ── Charts ───────────────────────────────────────────────────
const CHART_GRID   = "rgba(255,255,255,.05)";
const TICK_COLOR   = "#6b7280";
const TICK_FONT    = { size: 10, family: "'Space Mono', monospace" };

function buildHourlyData(airline, origin, dest) {
  const base = [0.18,0.14,0.12,0.11,0.12,0.15,0.22,0.31,0.38,0.35,0.33,0.3,0.35,0.4,0.43,0.42,0.41,0.46,0.49,0.51,0.48,0.42,0.35,0.25];
  const seed = (airline + origin + dest).split("").reduce((a, c) => a + c.charCodeAt(0), 0);
  return base.map((v, i) => Math.min(0.95, Math.max(0.05, v + ((seed * i * 7 + i * 13) % 100) / 700 - 0.07)));
}

let hourlyChart = null;
function renderHourlyChart(airline, origin, dest, currentHour) {
  const canvas = $("chart-hourly");
  if (!canvas || !window.Chart) return;

  // Ensure the canvas has real dimensions before rendering
  requestAnimationFrame(() => {
    const ctx = canvas.getContext("2d");
    const data = buildHourlyData(airline || "XX", origin || "XXX", dest || "XXX");
    const labels = Array.from({ length: 24 }, (_, i) => `${String(i).padStart(2, "0")}h`);
    const colors = data.map((_, i) => i === currentHour ? "#f7c94f" : "rgba(79,142,247,.65)");

    if (hourlyChart) { hourlyChart.destroy(); hourlyChart = null; }
    hourlyChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{ data: data.map((v) => +(v * 100).toFixed(1)), backgroundColor: colors, borderRadius: 3 }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        plugins: { legend: { display: false }, tooltip: {
          callbacks: { label: (ctx) => ` ${ctx.parsed.y}% delay chance` },
        }},
        scales: {
          x: { grid: { color: CHART_GRID }, ticks: { color: TICK_COLOR, font: TICK_FONT } },
          y: {
            grid: { color: CHART_GRID },
            ticks: { color: TICK_COLOR, font: TICK_FONT, callback: (v) => `${v}%` },
            min: 0, max: 80,
          },
        },
      },
    });
  });
}

const CARRIERS = ["G3","JJ","AD","LA","AA","DL","UA","KL","AF","IB"];
const CARRIER_NAMES = { G3:"Gol", JJ:"LATAM", AD:"Azul", LA:"LATAM", AA:"American", DL:"Delta", UA:"United", KL:"KLM", AF:"Air France", IB:"Iberia" };

function buildCarrierData(origin, dest, myAirline) {
  const seed = (origin + dest).split("").reduce((a, c) => a + c.charCodeAt(0), 0);
  const carriers = CARRIERS.filter((c) => c !== myAirline).slice(0, 5);
  if (myAirline) carriers.unshift(myAirline);
  return carriers.map((c, i) => ({
    airline: c,
    name: CARRIER_NAMES[c] || c,
    prob: Math.min(0.85, Math.max(0.1, 0.25 + ((seed * (i + 1) * 3) % 100) / 250)),
  }));
}

let carrierChart = null;
function renderCarrierChart(origin, dest, myAirline) {
  const wrap = $("carrier-wrap");
  if (!wrap || !window.Chart) return;
  wrap.innerHTML = '<canvas id="chart-carrier" style="width:100%;height:100%"></canvas>';

  requestAnimationFrame(() => {
    const canvas = $("chart-carrier");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const data = buildCarrierData(origin, dest, myAirline);
    const colors = data.map((d) => d.airline === myAirline ? "#f7c94f" : "rgba(79,247,160,.7)");

    if (carrierChart) { carrierChart.destroy(); carrierChart = null; }
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
        animation: { duration: 400 },
        plugins: { legend: { display: false }, tooltip: {
          callbacks: { label: (ctx) => ` ${ctx.parsed.x}% delay chance` },
        }},
        scales: {
          x: { grid: { color: CHART_GRID }, ticks: { color: TICK_COLOR, font: TICK_FONT, callback: (v) => `${v}%` }, max: 80 },
          y: { grid: { display: false }, ticks: { color: "#e8eaf0", font: { size: 11, family: "'DM Sans', sans-serif" } } },
        },
      },
    });
  });
}

// ── Factors bar list ─────────────────────────────────────────
function renderFactors(topFactors) {
  const wrap = $("radar-wrap");
  if (!wrap) return;
  if (!topFactors || !topFactors.length) {
    wrap.innerHTML = '<div class="empty-state"><div class="empty-icon">📡</div><div class="empty-text">Awaiting analysis</div></div>';
    return;
  }
  const parsed = topFactors.map((f) => ({
    name: f.feature.replace(/_/g, " "),
    val: parseFloat((f.impact || "").replace("%", "")) || 0,
  }));
  const maxAbs = Math.max(...parsed.map((p) => Math.abs(p.val)), 1);

  wrap.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:10px;padding:4px 0;height:100%;overflow-y:auto">
      ${parsed.map((p) => {
        const isPos = p.val > 0;
        const barW  = Math.min(100, (Math.abs(p.val) / maxAbs) * 100);
        const color = isPos ? "var(--danger,#f7604f)" : "var(--ok,#4ff7a0)";
        return `
          <div style="display:grid;grid-template-columns:1fr 2fr 48px;align-items:center;gap:10px">
            <div style="font-size:11px;font-family:'Space Mono',monospace;color:#9ca3af;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${p.name}</div>
            <div style="height:6px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${barW}%;background:${color};border-radius:3px;transition:width .5s"></div>
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:11px;font-weight:700;color:${color};text-align:right">${p.val > 0 ? "+" : ""}${p.val.toFixed(1)}%</div>
          </div>`;
      }).join("")}
    </div>`;
}

// ── Show result ───────────────────────────────────────────────
function showResult(data) {
  const prob  = Number(data.delay_probability || 0);
  const pct   = Math.round(prob * 100);
  const level = String(data.risk_level || "LOW").toUpperCase();

  const badge = $("risk-badge");
  if (badge) {
    const MAP = { LOW: "LOW RISK", MEDIUM: "MEDIUM RISK", HIGH: "HIGH RISK" };
    const BG  = { LOW: "rgba(79,247,160,.12)", MEDIUM: "rgba(247,201,79,.12)", HIGH: "rgba(247,96,79,.12)" };
    const CLR = { LOW: "#4ff7a0",              MEDIUM: "#f7c94f",              HIGH: "#f7604f" };
    badge.textContent    = MAP[level] || level;
    badge.style.background   = BG[level]  || "rgba(255,255,255,.06)";
    badge.style.color        = CLR[level] || "#e8eaf0";
    badge.style.borderColor  = CLR[level] || "rgba(255,255,255,.1)";
  }

  const probEl = $("prob-num");
  if (probEl) {
    probEl.textContent = String(pct);
    probEl.style.color = level === "HIGH" ? "#f7604f" : level === "MEDIUM" ? "#f7c94f" : "#4ff7a0";
  }

  const adviceEl = $("advice-inline");
  if (adviceEl) adviceEl.textContent = data.advice || "";
}

// ── Run analysis ──────────────────────────────────────────────
async function runAnalysis() {
  const origin  = $("f-origin").value.trim().toUpperCase();
  const dest    = $("f-dest").value.trim().toUpperCase();
  const airline = $("f-airline").value.trim().toUpperCase();
  const dep     = $("f-dep").value.trim();
  const date    = $("f-date").value;
  const dist    = $("f-dist").value;

  if (!origin || !dest || !airline || !dep) {
    alert("Please fill in Origin, Destination, Airline, and Departure Time.");
    return;
  }

  const btn  = $("btn-analyze");
  btn.disabled = true;
  btn.textContent = "Analyzing...";

  drawRoute(origin, dest);
  const hour = Math.floor(parseInt(dep, 10) / 100);
  renderHourlyChart(airline, origin, dest, Number.isNaN(hour) ? -1 : hour);
  renderCarrierChart(origin, dest, airline);

  const payload = { origin_airport: origin, destination_airport: dest, airline, scheduled_departure: parseInt(dep, 10) };
  if (date) payload.flight_date = date;
  if (dist) payload.distance    = parseFloat(dist);

  try {
    const r    = await fetch(apiBase() + "/advise", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: AbortSignal.timeout(10000) });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    state.analysisResult = data;
    showResult(data);
    renderFactors(data.top_factors);
  } catch {
    const mockProb = 0.34 + Math.random() * 0.3;
    const mockData = {
      delay_probability: mockProb,
      risk_level: mockProb >= 0.7 ? "HIGH" : mockProb >= 0.4 ? "MEDIUM" : "LOW",
      top_factors: [
        { feature: "CARRIER_DELAY_RATE",  impact: "+8.2%" },
        { feature: "ORIGIN_DELAY_RATE",   impact: "+5.1%" },
        { feature: "ROUTE_DELAY_RATE",    impact: "-2.3%" },
      ],
      advice: `[Demo] Route ${origin}→${dest} — simulated probability: ${(mockProb * 100).toFixed(0)}%.`,
    };
    showResult(mockData);
    renderFactors(mockData.top_factors);
  } finally {
    btn.disabled    = false;
    btn.textContent = "ANALYZE DELAY RISK";
  }
}

// ── Weekly predictions table ──────────────────────────────────
async function loadWeeklyPredictions() {
  const wrap     = $("weekly-predictions-wrap");
  const table    = $("weekly-predictions-table");
  const emptyEl  = wrap ? wrap.querySelector(".empty-state") : null;

  try {
    const response = await fetch(`${apiBase()}/api/upcoming_flights?limit=60`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    state.upcomingFlights = data.predictions || [];
    updateAvailableFlightsCatalog();

    if (data.predictions && data.predictions.length > 0) {
      populateTable(table, data.predictions);
      table.classList.remove("hidden");
      if (emptyEl) emptyEl.classList.add("hidden");
    } else {
      if (emptyEl) emptyEl.querySelector(".empty-text").textContent = data.detail || "No upcoming flights found.";
    }
  } catch {
    state.upcomingFlights = [];
    updateAvailableFlightsCatalog();
    if (emptyEl) emptyEl.querySelector(".empty-text").textContent = "Failed to load flight list.";
  }
}

function populateTable(table, predictions) {
  if (!table || !predictions.length) return;
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  const headers   = Object.keys(predictions[0]);
  const headerRow = document.createElement("tr");
  headers.forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  predictions.forEach((prediction) => {
    const row = document.createElement("tr");
    headers.forEach((h) => {
      const td = document.createElement("td");
      td.textContent = prediction[h];
      row.appendChild(td);
    });
    tbody.appendChild(row);
  });
}

// ── Init ──────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  const input = apiBaseInput();
  if (input) input.value = apiBase();
  $("f-date").value = new Date().toISOString().slice(0, 10);
  bindApiBaseField();
  bindAutocomplete();
  initMap();
  checkApiHealth();
  loadWeeklyPredictions();

  // Render the hourly chart after the layout has settled
  setTimeout(() => renderHourlyChart("XX", "XXX", "XXX", -1), 300);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && document.activeElement === $("flight-number-input")) lookupFlight();
  });
});
