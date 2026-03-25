function toNumberOrNull(rawValue) {
  if (rawValue === undefined || rawValue === null) return null;
  const value = String(rawValue).trim();
  if (value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function buildPayload(form) {
  const data = new FormData(form);
  return {
    origin_airport: String(data.get("origin_airport") || "").trim(),
    destination_airport: String(data.get("destination_airport") || "").trim(),
    airline: String(data.get("airline") || "").trim(),
    scheduled_departure: toNumberOrNull(data.get("scheduled_departure")),
    month: toNumberOrNull(data.get("month")),
    day_of_week: toNumberOrNull(data.get("day_of_week")),
    distance: toNumberOrNull(data.get("distance")),
    question: String(data.get("question") || "").trim() || null,
  };
}

function riskClass(riskLevel) {
  const level = String(riskLevel || "").toUpperCase();
  if (level === "LOW") return "LOW";
  if (level === "MEDIUM") return "MEDIUM";
  return "HIGH";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
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
      const probability = Number.isFinite(Number(flight.delay_probability))
        ? `<span class="advisor-flight-prob">${(Number(flight.delay_probability) * 100).toFixed(1)}% predicted risk</span>`
        : "";

      return `
        <article class="advisor-flight-card">
          <div class="advisor-flight-head">
            <strong>${escapeHtml(flightCode)}</strong>
            ${risk}
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

function renderResponse(target, data) {
  const adviceSource = String(data.advice_source || "heuristic");
  const adviceClass = adviceSource === "nvidia_nemotron" ? "ok" : "warning";
  const adviceLabel = adviceSource === "nvidia_nemotron" ? "Nemotron" : "Local fallback";
  const delayProbability = Number(data.delay_probability);
  const probabilityText = Number.isFinite(delayProbability) ? (delayProbability * 100).toFixed(1) : "0.0";

  target.classList.remove("empty", "error");
  target.innerHTML = `
    <div class="result-bar">
      <div class="prob-display">
        <strong class="prob-num">${probabilityText}</strong>
        <span class="prob-pct">% delay risk</span>
      </div>
      <span class="risk-badge ${riskClass(data.risk_level)}">${escapeHtml(data.risk_level || "HIGH")}</span>
      <span class="status-chip ${adviceClass}">${escapeHtml(adviceLabel)}</span>
      <div class="advice-inline">${escapeHtml(data.advice || "")}</div>
    </div>
    ${renderFactors(data)}
    ${renderSuggestedFlights(data)}
  `;
}

function renderError(target, message) {
  target.classList.remove("empty");
  target.classList.add("error");
  target.textContent = message || "Unexpected error while calling /advise.";
}

async function submitForm(form, resultEl) {
  const payload = buildPayload(form);
  if (!payload.origin_airport || !payload.destination_airport || !payload.airline) {
    renderError(resultEl, "Fill in origin, destination and airline.");
    return;
  }
  if (payload.scheduled_departure === null) {
    renderError(resultEl, "Scheduled departure must be a number (HHMM).");
    return;
  }

  const button = form.querySelector("button[type='submit']");
  if (button) {
    button.disabled = true;
    button.textContent = "Processing...";
  }

  try {
    const response = await fetch("/advise", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      renderError(resultEl, body.detail || `Request failed (${response.status}).`);
      return;
    }
    renderResponse(resultEl, body);
  } catch (error) {
    renderError(resultEl, error?.message || "Network error.");
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = form.id === "advisor-form" ? "Ask Advisor" : "Calculate";
    }
  }
}

function bindForm(formId, resultId) {
  const form = document.getElementById(formId);
  const result = document.getElementById(resultId);
  if (!form || !result) return;

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitForm(form, result);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  bindForm("prediction-form", "prediction-result");
  bindForm("advisor-form", "advisor-result");
});
