/**
 * live_map.js - Live flight map powered by Leaflet and the /api/live_flights endpoint.
 *
 * Dependencies:
 *   <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
 *   <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
 *
 * Minimum HTML:
 *   <div id="live-map" style="width:100%;height:600px;"></div>
 *   <div id="live-map-status"></div>
 *   <script src="/static/live_map.js"></script>
 */

const LIVE_FLIGHTS_URL = "/api/live_flights";
const REFRESH_INTERVAL_MS = 60_000;
const FETCH_TIMEOUT_MS = 15_000;
const MAX_ALTITUDE_FT = 45_000;

const map = L.map("live-map").setView([-15.0, -50.0], 4);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors',
  maxZoom: 18,
}).addTo(map);

function planeIcon(heading = 0) {
  return L.divIcon({
    className: "",
    html: `<div style="
      width:32px;
      height:32px;
      display:flex;
      align-items:center;
      justify-content:center;
      transform:rotate(${heading}deg);
      font-size:26px;
      line-height:1;
      filter:
        drop-shadow(0 0 3px rgba(0,0,0,0.7))
        drop-shadow(0 2px 4px rgba(0,0,0,0.5));
      transition:transform 0.4s ease;
    ">&#9992;</div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });
}

function climbIndicator(verticalRateFpm) {
  if (verticalRateFpm == null) return "Level";
  if (verticalRateFpm > 100) return `Climbing <small>(${verticalRateFpm.toLocaleString()} ft/min)</small>`;
  if (verticalRateFpm < -100) return `Descending <small>(${Math.abs(verticalRateFpm).toLocaleString()} ft/min)</small>`;
  return "Level";
}

function altitudeBar(altitudeFt) {
  if (altitudeFt == null) return "";
  const pct = Math.min(100, Math.round((altitudeFt / MAX_ALTITUDE_FT) * 100));
  const hue = Math.round(120 - pct * 1.2);
  return `
    <div style="margin-top:4px;">
      <div style="font-size:11px;color:#666;margin-bottom:2px;">
        Altitude: ${altitudeFt.toLocaleString()} ft (${pct}% of max)
      </div>
      <div style="background:#ddd;border-radius:4px;height:7px;width:100%;">
        <div style="
          width:${pct}%;
          height:7px;
          border-radius:4px;
          background:hsl(${hue}, 80%, 45%);
          transition:width 0.4s ease;
        "></div>
      </div>
    </div>`;
}

function flightAwareLink(callsign, icao24) {
  const id = callsign ? callsign.trim() : icao24;
  const flightAwareUrl = `https://flightaware.com/live/flight/${encodeURIComponent(id)}`;
  const flightradarUrl = `https://www.flightradar24.com/${encodeURIComponent(id)}`;
  return `
    <div style="margin-top:6px;display:flex;gap:8px;font-size:12px;">
      <a href="${flightAwareUrl}" target="_blank" rel="noopener" style="color:#1a73e8;">FlightAware</a>
      <a href="${flightradarUrl}" target="_blank" rel="noopener" style="color:#e8601a;">FlightRadar24</a>
    </div>`;
}

function buildPopup(flight) {
  const {
    icao24,
    heading,
    callsign,
    altitude_ft: altitudeFt,
    speed_kmh: speedKmh,
    origin_country: originCountry,
    vertical_rate_fpm: verticalRateFpm,
  } = flight;

  return `
    <div style="font-family:sans-serif;font-size:13px;min-width:200px;">
      <div style="font-size:15px;font-weight:700;margin-bottom:4px;">
        ${callsign || icao24}
      </div>
      <div>Country: ${originCountry || "-"}</div>
      <div>ICAO: <code>${icao24}</code></div>
      <div>Heading: ${heading ?? "-"}</div>
      <div>Speed: ${speedKmh != null ? `${speedKmh} km/h` : "-"}</div>
      <div style="margin-top:4px;">${climbIndicator(verticalRateFpm)}</div>
      ${altitudeBar(altitudeFt)}
      ${flightAwareLink(callsign, icao24)}
    </div>`;
}

const markers = {};

function updateMarker(flight) {
  const { icao24, latitude, longitude, heading } = flight;
  if (!latitude || !longitude) return;

  const popup = buildPopup(flight);

  if (markers[icao24]) {
    markers[icao24]
      .setLatLng([latitude, longitude])
      .setIcon(planeIcon(heading ?? 0))
      .getPopup()
      .setContent(popup);
    return;
  }

  markers[icao24] = L.marker([latitude, longitude], { icon: planeIcon(heading ?? 0) })
    .bindPopup(popup)
    .addTo(map);
}

function removeStaleMarkers(activeCodes) {
  const active = new Set(activeCodes);
  for (const icao24 of Object.keys(markers)) {
    if (!active.has(icao24)) {
      map.removeLayer(markers[icao24]);
      delete markers[icao24];
    }
  }
}

const statusEl = document.getElementById("live-map-status");

async function fetchAndUpdate() {
  if (statusEl) statusEl.textContent = "Updating...";

  try {
    const resp = await fetch(
      `${LIVE_FLIGHTS_URL}?region=brazil&limit=180&degraded=1`,
      { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) },
    );
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const data = await resp.json();
    if (statusEl && data.live_available === false) {
      statusEl.textContent = `Live tracking unavailable. ${data.detail || "Scheduled departures remain available."}`;
      return;
    }

    const flights = data.flights || [];
    flights.forEach(updateMarker);
    removeStaleMarkers(flights.map((flight) => flight.icao24));

    if (statusEl) {
      const ts = new Date().toLocaleTimeString("en-US");
      const staleLabel = data.stale_cache === true ? " - cached snapshot" : "";
      statusEl.textContent = `Updated ${flights.length} aircraft at ${ts} - cache ${data.cache_age_sec}s${staleLabel}`;
    }
  } catch (err) {
    console.error("Error fetching flights:", err);
    if (statusEl) {
      statusEl.textContent = `Live tracking error: ${err.message}. Showing the last available snapshot if present.`;
    }
  }
}

fetchAndUpdate();
setInterval(fetchAndUpdate, REFRESH_INTERVAL_MS);
