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
  return `${airport.iata_code} - ${parts.join(" / ") || "Aeroporto sem descricao"}`;
}

async function loadCountries() {
  if (!flightElements.country) return;
  setFlightFeedback("Carregando paises...");
  resetAirportSelect("Selecione um pais primeiro");
  resetTable("A tabela sera exibida apos selecionar um aeroporto.");

  try {
    const response = await fetch("/api/flight/countries");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Falha ao carregar paises (${response.status}).`);
    }

    const countries = Array.isArray(data.countries) ? data.countries : [];
    if (!countries.length) {
      setSelectOptions(flightElements.country, "Nenhum pais encontrado", [], () => "", () => "");
      setFlightFeedback("Nenhum pais com aeroporto disponivel foi encontrado.", true);
      return;
    }

    setSelectOptions(
      flightElements.country,
      "Selecione um pais",
      countries,
      (country) => country.country,
      buildCountryLabel
    );

    setFlightFeedback(
      `Base de aeroportos carregada: ${data.total_countries || countries.length} paises.`
    );
  } catch (error) {
    setSelectOptions(flightElements.country, "Falha ao carregar", [], () => "", () => "");
    setFlightFeedback(error?.message || "Erro ao buscar paises.", true);
  }
}

async function loadAirportsByCountry(country) {
  if (!country || !flightElements.airport) {
    resetAirportSelect("Selecione um pais primeiro");
    resetTable("A tabela sera exibida apos selecionar um aeroporto.");
    return;
  }

  setFlightFeedback(`Carregando aeroportos de ${country}...`);
  resetAirportSelect("Carregando aeroportos...");
  resetTable("Selecione um aeroporto para ver os voos locais.");

  try {
    const url = `/api/flight/airports?country=${encodeURIComponent(country)}`;
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Falha ao carregar aeroportos (${response.status}).`);
    }

    const airports = Array.isArray(data.airports) ? data.airports : [];
    if (!airports.length) {
      resetAirportSelect("Nenhum aeroporto encontrado para este pais");
      setFlightFeedback(`Nenhum aeroporto encontrado para ${country}.`, true);
      return;
    }

    flightElements.airport.disabled = false;
    setSelectOptions(
      flightElements.airport,
      "Selecione um aeroporto",
      airports,
      (airport) => airport.iata_code,
      buildAirportLabel
    );

    setFlightFeedback(`${airports.length} aeroportos encontrados em ${country}.`);
  } catch (error) {
    resetAirportSelect("Falha ao carregar aeroportos");
    setFlightFeedback(error?.message || "Erro ao buscar aeroportos.", true);
  }
}

async function loadAirportDepartures(iataCode) {
  if (!iataCode) {
    resetTable("A tabela sera exibida apos selecionar um aeroporto.");
    if (flightElements.summary) {
      flightElements.summary.textContent = "Selecione um aeroporto para carregar os voos.";
    }
    return;
  }

  resetTable("Carregando voos locais...");
  setFlightFeedback(`Consultando voos para ${iataCode}...`);
  if (flightElements.summary) {
    flightElements.summary.textContent = `Buscando voos de saida para ${iataCode}...`;
  }

  try {
    const url = `/api/flight/departures?airport=${encodeURIComponent(iataCode)}&limit=80`;
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `Falha ao carregar voos (${response.status}).`);
    }

    const departures = Array.isArray(data.departures) ? data.departures : [];
    if (!departures.length) {
      resetTable("Nenhum voo local futuro encontrado para este aeroporto.");
    } else {
      showTableRows(departures);
    }

    if (flightElements.summary) {
      flightElements.summary.textContent = `${data.returned_rows || departures.length} de ${
        data.matched_rows || departures.length
      } voos exibidos para ${iataCode}.`;
    }

    setFlightFeedback(
      data.future_window
        ? "Exibindo voos futuros a partir da data atual."
        : "Exibindo voos disponiveis na base."
    );
  } catch (error) {
    resetTable("Falha ao carregar voos locais.");
    setFlightFeedback(error?.message || "Erro ao buscar voos.", true);
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
