const DEFAULT_DISCOVERY_PROMPTS = [
  "I want to search for a trip to a specific country.",
  "What is the best day or time to fly with lower delay risk?",
  "When is the best time to book the ticket?",
];

const advisorState = {
  countries: null,
  countriesPromise: null,
  airportsByCountry: new Map(),
  sessionId: null,
  messages: [],
};

function toNumberOrNull(rawValue) {
  if (rawValue === undefined || rawValue === null) return null;
  const value = String(rawValue).trim();
  if (value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function toTextOrNull(rawValue) {
  const value = String(rawValue ?? "").trim();
  return value || null;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function normalizeErrorMessage(message) {
  if (Array.isArray(message)) return message.map((item) => JSON.stringify(item)).join(" | ");
  if (typeof message === "object" && message !== null) return JSON.stringify(message);
  return String(message || "Unexpected error while calling /advise.");
}

function truncateSelectLabel(rawValue, maxLength = 44) {
  const value = String(rawValue ?? "").trim();
  if (value.length <= maxLength) return value;
  return `${value.slice(0, Math.max(0, maxLength - 3)).trimEnd()}...`;
}

function riskClass(riskLevel) {
  const level = String(riskLevel || "").toUpperCase();
  if (level === "LOW") return "LOW";
  if (level === "MEDIUM") return "MEDIUM";
  return "HIGH";
}

function resolveDelayPrediction(rawValue, fallbackProbability = null) {
  if (rawValue === true) return 1;
  if (rawValue === false) return 0;

  const numeric = Number(rawValue);
  if (Number.isFinite(numeric)) {
    if (numeric === 1) return 1;
    if (numeric === 0) return 0;
  }

  const probability = Number(fallbackProbability);
  if (Number.isFinite(probability)) {
    return probability >= 0.5 ? 1 : 0;
  }
  return null;
}

function renderDelayPredictionChip(rawValue, fallbackProbability = null) {
  const prediction = resolveDelayPrediction(rawValue, fallbackProbability);
  if (prediction === null) return "";
  const chipClass = prediction === 1 ? "error" : "ok";
  const label = prediction === 1 ? "delay predicted" : "on-time predicted";
  return `<span class="status-chip ${chipClass}">${label}</span>`;
}

function isLlmAdviceSource(adviceSource) {
  const value = String(adviceSource || "").toLowerCase();
  return value === "nvidia" || value === "nvidia_nemotron" || value === "huggingface";
}

function buildPredictionPayload(form) {
  const data = new FormData(form);
  return {
    origin_airport: toTextOrNull(data.get("origin_airport")),
    destination_airport: toTextOrNull(data.get("destination_airport")),
    airline: toTextOrNull(data.get("airline")),
    scheduled_departure: toNumberOrNull(data.get("scheduled_departure")),
    month: toNumberOrNull(data.get("month")),
    day_of_week: toNumberOrNull(data.get("day_of_week")),
    distance: toNumberOrNull(data.get("distance")),
    question: toTextOrNull(data.get("question")),
  };
}

function buildAdvisorPayload(form) {
  const data = new FormData(form);
  return {
    origin_country: toTextOrNull(data.get("origin_country")),
    origin_airport: toTextOrNull(data.get("origin_airport")),
    destination_country: toTextOrNull(data.get("destination_country")),
    destination_airport: toTextOrNull(data.get("destination_airport")),
    airline: toTextOrNull(data.get("airline")),
    scheduled_departure: toNumberOrNull(data.get("scheduled_departure")),
    month: toNumberOrNull(data.get("month")),
    day_of_week: toNumberOrNull(data.get("day_of_week")),
    distance: toNumberOrNull(data.get("distance")),
    question: toTextOrNull(data.get("question")),
  };
}

function renderPromptButtons(prompts) {
  const items = Array.isArray(prompts) && prompts.length ? prompts : DEFAULT_DISCOVERY_PROMPTS;
  return items
    .map((prompt) => `
      <button type="button" class="advisor-prompt-btn" data-advisor-prompt="${escapeHtml(prompt)}">
        ${escapeHtml(prompt)}
      </button>
    `)
    .join("");
}

function renderFactors(data) {
  const factors = Array.isArray(data.top_factors) ? data.top_factors : [];
  if (!factors.length) return "";

  const items = factors
    .map((factor) => `
      <li>
        <strong>${escapeHtml(factor.feature)}</strong>
        <span class="mono">${escapeHtml(factor.impact)}</span>
      </li>
    `)
    .join("");

  return `
    <section class="advisor-section">
      <h3>Top factors</h3>
      <ul class="factor-list">${items}</ul>
    </section>
  `;
}

function renderSuggestedFlights(data) {
  const flights = Array.isArray(data.suggested_flights) ? data.suggested_flights : [];
  if (!flights.length) return "";

  const cards = flights
    .map((flight) => {
      const flightCode = [flight.airline, flight.flight_number].filter(Boolean).join("") || "Route option";
      const route = [flight.origin_airport, flight.destination_airport].filter(Boolean).join(" -> ");
      const schedule = [flight.flight_date_br || flight.flight_date, flight.scheduled_departure].filter(Boolean).join(" | ");
      const risk = flight.risk_level
        ? `<span class="risk-badge ${riskClass(flight.risk_level)}">${escapeHtml(flight.risk_level)}</span>`
        : "";
      const predictionChip = renderDelayPredictionChip(flight.delay_prediction, flight.delay_probability);
      const probability = Number.isFinite(Number(flight.delay_probability))
        ? `<span class="advisor-flight-prob">${(Number(flight.delay_probability) * 100).toFixed(1)}% predicted risk</span>`
        : "";

      return `
        <article class="advisor-flight-card">
          <div class="advisor-flight-head">
            <strong>${escapeHtml(flightCode)}</strong>
            <div class="advisor-flight-head-badges">${predictionChip}${risk}</div>
          </div>
          <div class="advisor-flight-meta">${escapeHtml(route || "Alternative route")}</div>
          <div class="advisor-flight-meta">${escapeHtml(schedule || "Scheduled flight")}</div>
          ${probability}
        </article>
      `;
    })
    .join("");

  return `
    <section class="advisor-section">
      <h3>Lower-risk scheduled options</h3>
      <div class="advisor-flight-grid">${cards}</div>
    </section>
  `;
}

function renderAssistantExtras(message) {
  const probability = Number(message.delay_probability);
  const probabilityMetric = Number.isFinite(probability)
    ? `<span class="advisor-inline-prob">${(probability * 100).toFixed(1)}%</span>`
    : "";
  const riskBadge = message.risk_level
    ? `<span class="risk-badge ${riskClass(message.risk_level)}">${escapeHtml(message.risk_level)}</span>`
    : "";
  const predictionChip = renderDelayPredictionChip(message.delay_prediction, message.delay_probability);
  const sourceChip = `
    <span class="status-chip ${isLlmAdviceSource(message.advice_source) ? "ok" : "warning"}">
      ${escapeHtml(message.advice_source || "assistant")}
    </span>
  `;
  const metrics = (probabilityMetric || riskBadge || predictionChip)
    ? `
      <div class="advisor-inline-metrics">
        ${probabilityMetric}
        ${predictionChip}
        ${riskBadge}
        ${sourceChip}
      </div>
    `
    : "";

  const prompts = Array.isArray(message.clarification_prompts) && message.clarification_prompts.length
    ? `
      <section class="advisor-section">
        <h3>Suggested next questions</h3>
        <div class="advisor-prompt-grid">${renderPromptButtons(message.clarification_prompts)}</div>
      </section>
    `
    : "";

  return `${metrics}${renderFactors(message)}${renderSuggestedFlights(message)}${prompts}`;
}

function renderAdvisorMessage(message) {
  const role = String(message.role || "assistant");
  const mode = message.mode ? `<span class="status-chip warning">${escapeHtml(message.mode)}</span>` : "";
  const timestamp = message.created_at ? escapeHtml(message.created_at.replace("T", " ")) : "";
  const extras = role === "assistant" ? renderAssistantExtras(message) : "";

  return `
    <article class="advisor-message ${role}">
      <div class="advisor-message-bubble">
        <div class="advisor-message-meta">
          <span>${role === "user" ? "Customer" : "Advisor"}</span>
          ${timestamp ? `<span>${timestamp}</span>` : ""}
          ${mode}
        </div>
        <div class="advisor-message-text">${escapeHtml(message.content || "")}</div>
        ${extras}
      </div>
    </article>
  `;
}

function renderAdvisorMessages(messages) {
  const target = document.getElementById("advisor-messages");
  if (!target) return;

  const safeMessages = Array.isArray(messages) ? messages : [];
  advisorState.messages = safeMessages;
  if (!safeMessages.length) {
    target.innerHTML = `<div class="advisor-chat-empty">No messages in this session yet.</div>`;
    return;
  }

  target.innerHTML = safeMessages.map(renderAdvisorMessage).join("");
  target.scrollTop = target.scrollHeight;
}

function updateAdvisorSessionMeta(sessionId) {
  advisorState.sessionId = sessionId || null;
  const target = document.getElementById("advisor-session-meta");
  if (!target) return;
  target.textContent = sessionId
    ? `Session ${sessionId.slice(0, 12)} • messages kept while this browser session is active`
    : "Session unavailable.";
}

function renderPredictionResponse(target, data) {
  const delayProbability = Number(data.delay_probability);
  const probabilityText = Number.isFinite(delayProbability) ? (delayProbability * 100).toFixed(1) : "0.0";
  const adviceSource = String(data.advice_source || "heuristic");
  const adviceClass = isLlmAdviceSource(adviceSource) ? "ok" : "warning";
  const predictionChip = renderDelayPredictionChip(data.delay_prediction, data.delay_probability);

  target.classList.remove("empty", "error");
  target.innerHTML = `
    <div class="result-bar">
      <div class="prob-display">
        <strong class="prob-num">${probabilityText}</strong>
        <span class="prob-pct">% delay risk</span>
      </div>
      ${predictionChip}
      <span class="risk-badge ${riskClass(data.risk_level)}">${escapeHtml(data.risk_level || "HIGH")}</span>
      <span class="status-chip ${adviceClass}">${escapeHtml(adviceSource)}</span>
      <div class="advice-inline">${escapeHtml(data.advice || "")}</div>
    </div>
    ${renderFactors(data)}
    ${renderSuggestedFlights(data)}
  `;
}

function renderPredictionError(target, message) {
  target.classList.remove("empty");
  target.classList.add("error");
  target.textContent = normalizeErrorMessage(message);
}

function setSelectOptions(selectEl, placeholder, items, valueFn, labelFn) {
  if (!selectEl) return;
  selectEl.innerHTML = "";

  const firstOption = document.createElement("option");
  firstOption.value = "";
  firstOption.textContent = truncateSelectLabel(placeholder, 52);
  firstOption.title = String(placeholder || "");
  selectEl.appendChild(firstOption);

  for (const item of items) {
    const option = document.createElement("option");
    option.value = valueFn(item);
    const label = String(labelFn(item) || "");
    option.textContent = truncateSelectLabel(label, 52);
    option.title = label;
    selectEl.appendChild(option);
  }
}

function buildCountryLabel(country) {
  const count = Number(country.airport_count || 0);
  return `${country.country} (${count})`;
}

function buildAirportLabel(airport) {
  const parts = [];
  if (airport.city) parts.push(airport.city);
  if (airport.airport_name) parts.push(airport.airport_name);
  return `${airport.iata_code} - ${parts.join(" / ") || "Airport without description"}`;
}

function getAdvisorLocationElements(role) {
  return {
    country: document.getElementById(`advisor-${role}-country`),
    airport: document.getElementById(`advisor-${role}-airport`),
  };
}

function setAdvisorHint(message, isError = false) {
  const hint = document.getElementById("advisor-location-feedback");
  if (!hint) return;
  hint.textContent = message;
  hint.classList.toggle("error", Boolean(isError));
}

function resetAirportSelect(selectEl, message) {
  if (!selectEl) return;
  selectEl.disabled = true;
  setSelectOptions(selectEl, message, [], () => "", () => "");
}

function clearAdvisorRouteContext(showHint = true) {
  const origin = getAdvisorLocationElements("origin");
  const destination = getAdvisorLocationElements("destination");

  if (origin.country) origin.country.value = "";
  if (destination.country) destination.country.value = "";
  if (origin.airport) resetAirportSelect(origin.airport, "Select the origin country first");
  if (destination.airport) resetAirportSelect(destination.airport, "Select the destination country first");

  for (const fieldName of ["airline", "scheduled_departure", "month", "day_of_week", "distance"]) {
    const field = document.querySelector(`#advisor-form [name='${fieldName}']`);
    if (field) field.value = "";
  }

  if (showHint) {
    setAdvisorHint("Route context reset.");
  }
}

function hasRouteDropdownValue(value) {
  return Boolean(value && (toTextOrNull(value.country) || toTextOrNull(value.airport)));
}

function questionMentionsRouteContext(question) {
  const text = String(question || "").trim().toLowerCase();
  if (!text) return false;
  return [
    /\b[a-z0-9]{3}\s*(?:-|->|>)\s*[a-z0-9]{3}\b/i,
    /\bsaindo\s+(?:de|do|da)\b/i,
    /\borigem\b/i,
    /\bdestino\b/i,
    /\bde\s+.+\s+(?:para|pra|pro)\s+/i,
    /\b(?:para|pra|pro)\s+[a-z]/i,
  ].some((pattern) => pattern.test(text));
}

function findLatestRouteUpdates(messages) {
  const items = Array.isArray(messages) ? messages : [];
  for (let index = items.length - 1; index >= 0; index -= 1) {
    const routeUpdates = items[index]?.route_updates;
    if (routeUpdates && (hasRouteDropdownValue(routeUpdates.origin) || hasRouteDropdownValue(routeUpdates.destination))) {
      return routeUpdates;
    }
  }
  return null;
}

async function applyAdvisorLocationUpdate(role, rawUpdate) {
  const update = rawUpdate || {};
  const nextCountry = toTextOrNull(update.country);
  const nextAirport = toTextOrNull(update.airport);
  if (!nextCountry && !nextAirport) return;

  const { country, airport } = getAdvisorLocationElements(role);
  if (!country || !airport) return;

  await ensureCountryOptions();

  if (!nextCountry) {
    if (nextAirport && Array.from(airport.options).some((option) => option.value === nextAirport)) {
      airport.value = nextAirport;
    }
    return;
  }

  if (!Array.from(country.options).some((option) => option.value === nextCountry)) {
    return;
  }

  const previousCountry = country.value || "";
  country.value = nextCountry;

  const airportAlreadyAvailable = nextAirport
    ? Array.from(airport.options).some((option) => option.value === nextAirport)
    : false;
  if (previousCountry !== nextCountry || airport.disabled || (nextAirport && !airportAlreadyAvailable)) {
    await loadAirportsByCountry(role, nextCountry);
  }

  if (nextAirport) {
    airport.value = Array.from(airport.options).some((option) => option.value === nextAirport) ? nextAirport : "";
  } else if (previousCountry !== nextCountry) {
    airport.value = "";
  }
}

async function applyAdvisorRouteUpdates(routeUpdates) {
  if (!routeUpdates) return;
  if (!hasRouteDropdownValue(routeUpdates.origin) && !hasRouteDropdownValue(routeUpdates.destination)) return;

  await applyAdvisorLocationUpdate("origin", routeUpdates.origin);
  await applyAdvisorLocationUpdate("destination", routeUpdates.destination);
  setAdvisorHint("Route filters updated from the conversation.");
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed (${response.status}).`);
  }
  return data;
}

function populateCountrySelects(countries) {
  for (const role of ["origin", "destination"]) {
    const { country } = getAdvisorLocationElements(role);
    if (!country) continue;
    setSelectOptions(country, "Select a country", countries, (item) => item.country, buildCountryLabel);
  }
}

async function ensureCountryOptions() {
  if (advisorState.countries) {
    populateCountrySelects(advisorState.countries);
    return;
  }

  if (!advisorState.countriesPromise) {
    advisorState.countriesPromise = fetchJson("/api/flight/countries")
      .then((data) => {
        advisorState.countries = Array.isArray(data.countries) ? data.countries : [];
        populateCountrySelects(advisorState.countries);
        setAdvisorHint("Select a country before the airport to avoid overloading the list.");
      })
      .catch((error) => {
        setAdvisorHint(error?.message || "Failed to load country options.", true);
      })
      .finally(() => {
        advisorState.countriesPromise = null;
      });
  }

  await advisorState.countriesPromise;
}

async function loadAirportsByCountry(role, country) {
  const { airport } = getAdvisorLocationElements(role);
  const countryLabel = role === "origin" ? "origin" : "destination";
  if (!airport) return;

  if (!country) {
    resetAirportSelect(airport, `Select the ${countryLabel} country first`);
    setAdvisorHint("Select a country before the airport to avoid overloading the list.");
    return;
  }

  resetAirportSelect(airport, "Loading airports...");
  setAdvisorHint(`Loading ${country} airports...`);

  try {
    let airports = advisorState.airportsByCountry.get(country);
    if (!airports) {
      const data = await fetchJson(`/api/flight/airports?country=${encodeURIComponent(country)}&limit=500`);
      airports = Array.isArray(data.airports) ? data.airports : [];
      advisorState.airportsByCountry.set(country, airports);
    }

    if (!airports.length) {
      resetAirportSelect(airport, "No airport found for this country");
      setAdvisorHint(`No airports available for ${country}.`, true);
      return;
    }

    airport.disabled = false;
    setSelectOptions(airport, "Select an airport", airports, (item) => item.iata_code, buildAirportLabel);
    setAdvisorHint(`${airports.length} airports loaded for ${country}.`);
  } catch (error) {
    resetAirportSelect(airport, "Failed to load airports");
    setAdvisorHint(error?.message || "Failed to load airports.", true);
  }
}

async function loadAdvisorHistory() {
  try {
    const data = await fetchJson("/api/advisor/history");
    updateAdvisorSessionMeta(data.session_id);
    renderAdvisorMessages(data.messages);
  } catch (error) {
    renderAdvisorMessages([{
      role: "assistant",
      content: normalizeErrorMessage(error?.message || "Failed to load chat history."),
      created_at: new Date().toISOString(),
      mode: "discovery",
      clarification_prompts: DEFAULT_DISCOVERY_PROMPTS,
    }]);
  }
}

async function resetAdvisorSession() {
  const button = document.getElementById("advisor-reset");
  if (button) {
    button.disabled = true;
    button.textContent = "Resetting...";
  }

  try {
    const data = await fetchJson("/api/advisor/reset", { method: "POST" });
    updateAdvisorSessionMeta(data.session_id);
    renderAdvisorMessages(data.messages);
    clearAdvisorRouteContext();
    const questionField = document.getElementById("advisor-question");
    if (questionField) questionField.value = "";
  } catch (error) {
    renderAdvisorMessages([{
      role: "assistant",
      content: normalizeErrorMessage(error?.message || "Failed to reset chat session."),
      created_at: new Date().toISOString(),
      mode: "discovery",
      clarification_prompts: DEFAULT_DISCOVERY_PROMPTS,
    }]);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "New Session";
    }
  }
}

async function submitPredictionForm(form, resultEl) {
  const payload = buildPredictionPayload(form);
  if (!payload.origin_airport || !payload.destination_airport) {
    renderPredictionError(resultEl, "Fill in origin and destination.");
    return;
  }

  const button = form.querySelector("button[type='submit']");
  if (button) {
    button.disabled = true;
    button.textContent = "Processing...";
  }

  try {
    const body = await fetchJson("/advise", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderPredictionResponse(resultEl, body);
  } catch (error) {
    renderPredictionError(resultEl, error?.message || "Network error.");
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "Calculate";
    }
  }
}

async function submitAdvisorForm(form) {
  const payload = buildAdvisorPayload(form);
  const button = form.querySelector("button[type='submit']");
  if (button) {
    button.disabled = true;
    button.textContent = "Sending...";
  }

  try {
    const body = await fetchJson("/advise", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    updateAdvisorSessionMeta(body.session_id);
    renderAdvisorMessages(body.messages);
    if (questionMentionsRouteContext(payload.question)) {
      clearAdvisorRouteContext(false);
    }
    await applyAdvisorRouteUpdates(body.route_updates || findLatestRouteUpdates(body.messages));
    const questionField = document.getElementById("advisor-question");
    if (questionField) {
      questionField.value = "";
      questionField.focus();
    }
  } catch (error) {
    renderAdvisorMessages([
      ...advisorState.messages,
      {
        role: "assistant",
        content: normalizeErrorMessage(error?.message || "Network error."),
        created_at: new Date().toISOString(),
        mode: "discovery",
        clarification_prompts: DEFAULT_DISCOVERY_PROMPTS,
      },
    ]);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "Send Message";
    }
  }
}

function bindPredictionForm() {
  const form = document.getElementById("prediction-form");
  const result = document.getElementById("prediction-result");
  if (!form || !result) return;

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitPredictionForm(form, result);
  });
}

function bindAdvisorForm() {
  const form = document.getElementById("advisor-form");
  if (!form) return;

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitAdvisorForm(form);
  });
}

function bindPromptButtons() {
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-advisor-prompt]");
    if (!button) return;

    const questionField = document.getElementById("advisor-question");
    const form = document.getElementById("advisor-form");
    const prompt = button.getAttribute("data-advisor-prompt") || "";
    if (!questionField || !form) return;

    questionField.value = prompt;
    form.requestSubmit();
  });
}

function bindAdvisorReset() {
  const button = document.getElementById("advisor-reset");
  if (!button) return;
  button.addEventListener("click", () => {
    resetAdvisorSession();
  });
}

async function initAdvisorLocationSelectors() {
  const origin = getAdvisorLocationElements("origin");
  const destination = getAdvisorLocationElements("destination");
  if (!origin.country || !origin.airport || !destination.country || !destination.airport) {
    return;
  }

  await ensureCountryOptions();
  resetAirportSelect(origin.airport, "Select the origin country first");
  resetAirportSelect(destination.airport, "Select the destination country first");

  origin.country.addEventListener("change", () => {
    loadAirportsByCountry("origin", origin.country.value);
  });

  destination.country.addEventListener("change", () => {
    loadAirportsByCountry("destination", destination.country.value);
  });
}

async function initAdvisorPage() {
  if (!document.getElementById("advisor-form")) return;
  await initAdvisorLocationSelectors();
  clearAdvisorRouteContext(false);
  await loadAdvisorHistory();
}

document.addEventListener("DOMContentLoaded", () => {
  bindPredictionForm();
  bindAdvisorForm();
  bindPromptButtons();
  bindAdvisorReset();
  initAdvisorPage();
});
