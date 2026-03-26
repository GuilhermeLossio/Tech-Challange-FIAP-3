const flightElements = {
  country: document.getElementById("flight-country-select"),
  airport: document.getElementById("flight-airport-select"),
  feedback: document.getElementById("flight-feedback"),
  summary: document.getElementById("flight-summary"),
  table: document.getElementById("flight-table"),
  tbody: document.querySelector("#flight-table tbody"),
  empty: document.getElementById("flight-empty"),
};

function setFlightFeedback(message, isError = false) {
  if (!flightElements.feedback) return;
  flightElements.feedback.textContent = message;
  flightElements.feedback.classList.toggle("flight-feedback-error", isError);
}

function setSelectOptions(selectEl, placeholder, items, valueFn, labelFn) {
  if (!selectEl) return;
  selectEl.innerHTML = "";

  const firstOption = document.createElement("option");
  firstOption.value = "";
  firstOption.textContent = placeholder;
  selectEl.appendChild(firstOption);

  for (const item of items) {
    const option = document.createElement("option");
    option.value = valueFn(item);
    option.textContent = labelFn(item);
    selectEl.appendChild(option);
  }
}

function resetAirportSelect(message) {
  if (!flightElements.airport) return;
  flightElements.airport.disabled = true;
  setSelectOptions(flightElements.airport, message, [], () => "", () => "");
}

function resetTable(message) {
  if (!flightElements.table || !flightElements.tbody || !flightElements.empty) return;
  flightElements.tbody.innerHTML = "";
  flightElements.table.classList.add("hidden");
  flightElements.empty.classList.remove("hidden");
  flightElements.empty.textContent = message;
}

function showTableRows(rows) {
  if (!flightElements.table || !flightElements.tbody || !flightElements.empty) return;
  const html = rows
    .map(
      (flight) => `
        <tr>
          <td>${flight.flight_date_br || "-"}</td>
          <td>${flight.airline || "-"}</td>
          <td>${flight.flight_code || "-"}</td>
          <td>${flight.origin_airport || "-"}</td>
          <td>${flight.destination_airport || "-"}</td>
          <td>${flight.scheduled_departure || "-"}</td>
          <td>${flight.scheduled_arrival || "-"}</td>
        </tr>
      `
    )
    .join("");

  flightElements.tbody.innerHTML = html;
  flightElements.empty.classList.add("hidden");
  flightElements.table.classList.remove("hidden");
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

async function loadCountries() {
  if (!flightElements.country) return;
  setFlightFeedback("Loading countries...");
  resetAirportSelect("Select a country first");
  resetTable("The table will appear after you select an airport.");

  try {
    const response = await fetch("/api/flight/countries");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Failed to load countries (${response.status}).`);
    }

    const countries = Array.isArray(data.countries) ? data.countries : [];
    if (!countries.length) {
      setSelectOptions(flightElements.country, "No country found", [], () => "", () => "");
      setFlightFeedback("No country with available airports was found.", true);
      return;
    }

    setSelectOptions(
      flightElements.country,
      "Select a country",
      countries,
      (country) => country.country,
      buildCountryLabel
    );

    setFlightFeedback(
      `Airport base loaded: ${data.total_countries || countries.length} countries.`
    );
  } catch (error) {
    setSelectOptions(flightElements.country, "Failed to load", [], () => "", () => "");
    setFlightFeedback(error?.message || "Error fetching countries.", true);
  }
}

async function loadAirportsByCountry(country) {
  if (!country || !flightElements.airport) {
    resetAirportSelect("Select a country first");
    resetTable("The table will appear after you select an airport.");
    return;
  }

  setFlightFeedback(`Loading airports for ${country}...`);
  resetAirportSelect("Loading airports...");
  resetTable("Select an airport to see local flights.");

  try {
    const url = `/api/flight/airports?country=${encodeURIComponent(country)}`;
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Failed to load airports (${response.status}).`);
    }

    const airports = Array.isArray(data.airports) ? data.airports : [];
    if (!airports.length) {
      resetAirportSelect("No airport found for this country");
      setFlightFeedback(`No airport found for ${country}.`, true);
      return;
    }

    flightElements.airport.disabled = false;
    setSelectOptions(
      flightElements.airport,
      "Select an airport",
      airports,
      (airport) => airport.iata_code,
      buildAirportLabel
    );

    setFlightFeedback(`${airports.length} airports found in ${country}.`);
  } catch (error) {
    resetAirportSelect("Failed to load airports");
    setFlightFeedback(error?.message || "Error fetching airports.", true);
  }
}

async function loadAirportDepartures(iataCode) {
  if (!iataCode) {
    resetTable("The table will appear after you select an airport.");
    if (flightElements.summary) {
      flightElements.summary.textContent = "Select an airport to load flights.";
    }
    return;
  }

  resetTable("Loading local flights...");
  setFlightFeedback(`Consultando voos para ${iataCode}...`);
  if (flightElements.summary) {
    flightElements.summary.textContent = `Searching departures for ${iataCode}...`;
  }

  try {
    const url = `/api/flight/departures?airport=${encodeURIComponent(iataCode)}&limit=80`;
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Failed to load flights (${response.status}).`);
    }

    const departures = Array.isArray(data.departures) ? data.departures : [];
    if (!departures.length) {
      resetTable("No upcoming local flight found for this airport.");
    } else {
      showTableRows(departures);
    }

    if (flightElements.summary) {
      flightElements.summary.textContent = `${data.returned_rows || departures.length} de ${
        data.matched_rows || departures.length
      } flights shown for ${iataCode}.`;
    }

    setFlightFeedback(
      data.future_window
        ? "Showing upcoming flights from the current date."
        : "Showing flights available in the dataset."
    );
  } catch (error) {
    resetTable("Failed to load local flights.");
    setFlightFeedback(error?.message || "Error fetching flights.", true);
  }
}

function bindFlightEvents() {
  if (!flightElements.country || !flightElements.airport) return;

  flightElements.country.addEventListener("change", () => {
    const country = flightElements.country.value;
    loadAirportsByCountry(country);
  });

  flightElements.airport.addEventListener("change", () => {
    const iataCode = flightElements.airport.value;
    loadAirportDepartures(iataCode);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  if (!flightElements.country) return;
  bindFlightEvents();
  loadCountries();
});
