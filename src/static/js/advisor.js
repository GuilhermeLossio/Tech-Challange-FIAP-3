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
  if (level === "LOW") return "risk-low";
  if (level === "MEDIUM") return "risk-medium";
  return "risk-high";
}

function renderResponse(target, data) {
  const factors = Array.isArray(data.top_factors) ? data.top_factors : [];
  const factorsHtml = factors
    .map((factor) => `<li><strong>${factor.feature}</strong> <span class="mono">${factor.impact}</span></li>`)
    .join("");

  target.classList.remove("empty", "error");
  target.innerHTML = `
    <div class="result-header">
      <strong>${(Number(data.delay_probability) * 100).toFixed(1)}%</strong>
      <span class="badge ${riskClass(data.risk_level)}">${data.risk_level}</span>
    </div>
    <p>${data.advice || ""}</p>
    ${factorsHtml ? `<ul class="factor-list">${factorsHtml}</ul>` : ""}
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
