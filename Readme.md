# Flight Advisor

Flight Advisor is a Flask application for flight-delay prediction and conversational travel assistance. The current codebase combines trained delay models, weekly route estimation, airport and route APIs, live-flight lookups, and a session-aware advisor UI.

## What the app currently includes

### Web pages
- `/` or `/front`: dashboard landing page
- `/flight` or `/flights`: flight lookup and route selection
- `/predictions`: prediction-focused page
- `/advisor`: conversational advisor
- `/dashboard`: optional mounted Dash app when `ENABLE_DASH=1` and Dash dependencies are available

### API and docs
- `/docs`, `/redoc`, `/openapi.json`
- `/health`
- `/predict`
- `/advise`
- `/api/advisor/history`
- `/api/advisor/reset`
- `/api/flight/countries`
- `/api/flight/airports`
- `/api/flight/departures`
- `/api/upcoming_flights`
- `/api/weekly_predictions`
- `/api/live_flights`
- `/api/live_flights/<icao24>`
- `/api/routes`

## Core capabilities

- Predict delay probability and a binary delay decision for a specific flight.
- Fall back to weekly route estimation when origin and destination are known but airline, date, or departure time are missing.
- Extract route context from natural-language questions and return `route_updates` so the frontend can sync the country and airport dropdowns automatically.
- Infer distance from the request, the historical route average, or the global average when explicit airline or country data is missing.
- Keep advisor chat history per session and expose history and reset endpoints.
- Use a configurable LLM provider for discovery, destination guidance, and complete travel-guide responses.
- Return model-based fallback advice even when the LLM is disabled or unavailable.
- Serve live-flight snapshots through OpenSky and upcoming schedule suggestions from the generated weekly dataset.

## Architecture snapshot

- Delivery layer: deployment entrypoint in `src/app.py`, Flask app composition in `src/api/main.py`, Jinja templates in `src/templates`, and static assets in `src/static`.
- View registration: `src/api/views/pages.py`, `src/api/views/flight.py`, and `src/api/views/advisor.py`.
- Prediction layer: model artifacts in `models/`, feature construction and fallback logic in `src/api/main.py`.
- Advisor layer: session-aware orchestration, route extraction, weekly fallback, and LLM prompt assembly.
- LLM layer: `src/api/services/llm_service.py` with provider selection for NVIDIA or Hugging Face compatible backends.
- Batch and jobs layer: future schedule generation and weekly prediction helpers in `src/jobs/`.
- Optional analytics layer: legacy or supplemental Dash app in `dashboard/app.py`.

## Repository structure

```text
FIAP-3/
|-- dashboard/                 # Optional Dash app and dashboard utilities
|-- data/                      # Raw, processed, and runtime data
|-- docs/                      # Mermaid and SVG architecture assets
|-- models/                    # Trained model artifacts and explainability exports
|-- notebooks/                 # Exploration and experimentation notebooks
|-- src/
|   |-- app.py                 # Deployment entrypoint for Railway or gunicorn
|   |-- api/
|   |   |-- main.py            # Flask app, schemas, predictors, API registration
|   |   |-- services/          # LLM, flight, and live-flight integrations
|   |   `-- views/             # Page, advisor, and flight endpoints
|   |-- jobs/                  # Weekly schedule and prediction jobs
|   |-- static/                # Browser JavaScript and CSS
|   |-- templates/             # Jinja templates
|   `-- rag/                   # Reserved or legacy area for retrieval artifacts
|-- PROTOTYPE.md               # Current technical prototype reference
|-- Readme.md                  # Project overview and setup
`-- requirements.txt
```

## Getting started

### Prerequisites

- Python 3.11
- A valid model artifact under `models/`
- Optional LLM credentials if you want advisor generation beyond heuristic fallback

### Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Create or update `.env`

The project loads environment variables automatically from `.env`. There is no `.env.example` in the repository at the moment, so create the file manually if needed.

Key variables:

| Variable | Purpose |
|---|---|
| `API_PORT` | Local Flask port. Defaults to `8000`. |
| `FLASK_DEBUG` | Set to `1` for debug mode. |
| `FLASK_SECRET_KEY` | Flask session secret. |
| `ENABLE_DASH` | Mount the Dash app under `/dashboard` when available. |
| `ADVISOR_LLM_ENABLED` | Enable or disable LLM calls for the advisor. |
| `ADVISOR_LLM_PROVIDER` or `LLM_PROVIDER` | LLM backend selection. Supports `nvidia` and `huggingface`. |
| `ADVISOR_LLM_MODEL` or `LLM_MODEL` | Shared model identifier. |
| `NVIDIA_API_KEY` or `NEMOTRON_API_KEY` | API key for NVIDIA-compatible calls. |
| `HF_TOKEN` or `HUGGINGFACE_API_KEY` | API key for Hugging Face router calls. |
| `ADVISOR_LLM_COMPACT_MODE` | Force compact responses for lightweight runs. |
| `QWEN_MAX_TOKENS` | Compact token ceiling used for Qwen-like models. |
| `ADVISOR_LLM_GUIDE_MAX_TOKENS` | Larger response budget for complete travel guides. |
| `ADVISOR_WEEKLY_WINDOW_DAYS` | Weekly prediction window size, clamped to `1..14`. |
| `AIRPORTS_INDEX_SOURCE` or `WORLD_AIRPORTS_SOURCE` | Airports source used by the flight dropdown APIs. |

### Run the current web app and API

```bash
python src/app.py
```

Then open:

- `http://localhost:8000/`
- `http://localhost:8000/advisor`
- `http://localhost:8000/docs`

### Run the Dash app standalone

If you still use the separate Dash dashboard directly:

```bash
python dashboard/app.py
```

### Railway or gunicorn

```bash
gunicorn -w 2 -b 0.0.0.0:$PORT src.app:app
```

If you prefer plain Python on Railway, `src/app.py` also reads `PORT` automatically:

```bash
python src/app.py
```

## Main API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |
| `POST` | `/predict` | Structured delay prediction |
| `POST` | `/advise` | Advisor workflow with prediction, weekly fallback, and LLM guidance |
| `GET` | `/api/advisor/history` | Load the current advisor chat session |
| `POST` | `/api/advisor/reset` | Clear chat history and route context |
| `GET` | `/api/flight/countries` | Countries available in the airports index |
| `GET` | `/api/flight/airports?country=...` | Airports for a selected country |
| `GET` | `/api/flight/departures?airport=...` | Upcoming departures or generated placeholders |
| `GET` | `/api/upcoming_flights` | Upcoming schedule view |
| `GET` | `/api/weekly_predictions` | Weekly prediction listing alias |
| `GET` | `/api/live_flights` | Live flights from OpenSky |
| `GET` | `/api/live_flights/<icao24>` | Live details for one aircraft |
| `GET` | `/api/routes` | Route and endpoint discovery payload |

## Advisor behavior

### Request model

`/advise` accepts structured fields plus a free-form question. The current request schema includes:

- `origin_country`
- `origin_airport`
- `destination_country`
- `destination_airport`
- `airline`
- `scheduled_departure`
- `flight_date`
- `year`, `month`, `day`, `day_of_week`
- `distance`
- `question`

### Response highlights

The advisor can return:

- `delay_probability`
- `delay_prediction`
- `risk_level`
- `top_factors`
- `suggested_flights`
- `clarification_prompts`
- `route_updates`
- `messages`
- `mode`
- `advice_source`
- `advice_model`

### Current routing and fallback rules

- If the user mentions a new origin or destination in natural language, the backend extracts it and sends `route_updates` so the frontend can sync the dropdowns.
- If a specific airport is detected, the corresponding country is filled automatically when possible.
- If origin and destination are known but airline, departure time, or exact date are missing, the advisor can use the upcoming weekly schedule instead of blocking on missing fields.
- If `distance` is missing, the predictor falls back to the historical route average or the global average.
- If the LLM is turned off or unavailable, the backend still returns deterministic model-based advice text.
- Session history is stored under `data/runtime/advisor_sessions`.

### Example request

```bash
curl -X POST "http://localhost:8000/advise" \
  -H "Content-Type: application/json" \
  -d '{
    "origin_airport": "GRU",
    "destination_airport": "JFK",
    "question": "Use the weekly schedule and tell me if this route is likely to be delayed."
  }'
```

### Example response shape

```json
{
  "delay_probability": 0.38,
  "delay_prediction": 0,
  "risk_level": "LOW",
  "mode": "weekly_route",
  "advice_source": "weekly_model",
  "top_factors": [
    {
      "feature": "distance",
      "impact": "Estimated distance from the historical average for this route."
    }
  ],
  "route_updates": {
    "origin": { "country": "Brazil", "airport": "GRU" },
    "destination": { "country": "United States", "airport": "JFK" }
  },
  "advice": "On-time predicted. The estimate uses the upcoming weekly schedule because no exact date was specified."
}
```

## Data and jobs

### Main artifacts

- `models/delay_model.pkl`: serialized delay model
- `models/delay_model_meta.json`: metadata for the prediction pipeline
- `models/explain/`: SHAP exports used by the explainability flow

### Supporting jobs

- `src/jobs/generate_future_flights.py`: future schedule generation
- `src/jobs/weekly_predict.py`: weekly prediction outputs
- `src/jobs/weekly_pipeline.py`: weekly processing flow
- `src/jobs/csv_to_parquet_converter.py`: CSV to Parquet helper

### Data dependencies

The app expects:

- a training or processed flight dataset for the model pipeline
- an airports index for country and airport dropdowns
- future or weekly schedule data for weekly route estimation
- OpenSky access for live-flight endpoints

## Limitations and next steps

- Real-time booking and purchase execution are not implemented in this backend.
- Live availability and fare shopping depend on external integrations that are still optional or disabled.
- The `dashboard/` app and the Flask pages overlap in some analytical capabilities and should be consolidated further.
- `src/rag/` is not the active runtime path for the current advisor flow and may be cleaned up or formalized later.

## Related docs

- `PROTOTYPE.md`: technical current-state reference
- `docs/`: diagrams and SVG assets used during design and presentation
