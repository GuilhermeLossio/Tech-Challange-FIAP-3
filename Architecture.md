# Flight Advisor — Architecture

> This document describes the current runtime architecture. For the live API docs see [Flight Advisor API Docs](https://flightadvisor.up.railway.app/docs/). For deployment and configuration see [`PROTOTYPE.md`](./PROTOTYPE.md).

---

## 1. System Overview

Flight Advisor is structured in four horizontal layers:

```
┌─────────────────────────────────────────────────────┐
│                  Delivery Layer                     │
│   Flask pages · Jinja2 templates · Static assets   │
├─────────────────────────────────────────────────────┤
│                    API Layer                        │
│   View registration · Request schemas · Routing    │
├─────────────────────────────────────────────────────┤
│                 Intelligence Layer                  │
│   Delay model · Weekly estimator · LLM service     │
├─────────────────────────────────────────────────────┤
│                    Data Layer                       │
│   Model artifacts · Airports index · Session store │
│   OpenSky · Weekly schedule · SHAP exports         │
└─────────────────────────────────────────────────────┘
```

---

## 2. Component Map

```mermaid
flowchart TB
    U[User] --> P[Flask Pages\nsrc/templates + src/static]
    P --> A[Flask API\nsrc/api/main.py]

    subgraph views[View Registration]
        V1[pages.py]
        V2[flight.py]
        V3[advisor.py]
    end

    subgraph intelligence[Intelligence Layer]
        M[Delay Model\ndelay_model.pkl]
        W[Weekly Estimator\nupcoming schedule]
        L[LLM Service\nNVIDIA · Hugging Face]
    end

    subgraph data[Data Layer]
        S[Session Store\ndata/runtime/advisor_sessions]
        X[Airports Index]
        O[OpenSky Service]
        Sh[SHAP Exports\nmodels/explain/]
    end

    A --> views
    A --> intelligence
    A --> data

    M --> R[Advisor Response]
    W --> R
    L --> R
    S --> R
    X --> R
    O --> R

    R --> P --> U
```

---

## 3. Advisor Request Lifecycle

```mermaid
sequenceDiagram
    actor U as User
    participant UI as Advisor Page
    participant A as POST /advise
    participant NLP as Route Extraction
    participant S as Session Store
    participant M as Delay Model
    participant W as Weekly Estimator
    participant L as LLM Service

    U->>UI: Enter route + question
    UI->>A: POST structured payload

    A->>NLP: Parse question for route mentions
    NLP-->>A: Detected airports · countries

    A->>S: Load session history

    alt Full flight context
        A->>M: Run specific-flight prediction
        M-->>A: probability · prediction · top_factors
    else Route known, details missing
        A->>W: Estimate from weekly schedule
        W-->>A: aggregate probability · suggested_flights
    else Open discovery question
        A->>A: Build discovery / clarification response
    end

    opt LLM enabled
        A->>L: Send structured advisor context
        L-->>A: Generated advice text
    end

    A->>S: Save updated message history
    A-->>UI: advice · metrics · route_updates · messages
    UI-->>U: Sync dropdowns · render response
```

---

## 4. Prediction Decision Tree

```mermaid
flowchart TD
    Q[Incoming /advise request] --> C1{origin + destination\nknown?}

    C1 -- No --> D[Discovery mode\nreturn clarification_prompts]

    C1 -- Yes --> C2{airline + date +\ndeparture known?}

    C2 -- Yes --> SF[Specific-flight prediction\ndelay_model.pkl]
    SF --> P1[delay_probability\ndelay_prediction\nrisk_level\ntop_factors]

    C2 -- No --> WR[Weekly route estimation\nupcoming schedule]
    WR --> P2[aggregate probability\nsuggested_flights\nmode = weekly_route]

    P1 --> LLM{LLM enabled?}
    P2 --> LLM
    D --> LLM

    LLM -- Yes --> G[Generate advice text\nvia LLM service]
    LLM -- No --> H[Deterministic heuristic\nfallback text]

    G --> R[Final /advise response]
    H --> R
```

---

## 5. Missing-Feature Fallback Chain

When the advisor has enough route context to estimate delay, it resolves missing predictive inputs instead of failing hard:

```mermaid
flowchart LR
    A[Delay assessment request] --> R{origin + destination\navailable?}

    R -- No --> D[Discovery mode\nask for missing airports]
    R -- Yes --> DIST{distance provided?}
    DIST -- Yes --> DIST1[Use request distance]
    DIST -- No --> DIST2{route average available?}
    DIST2 -- Yes --> DIST3[Use route average]
    DIST2 -- No --> DIST4[Use global average distance]

    DIST1 --> AIR{airline provided?}
    DIST3 --> AIR
    DIST4 --> AIR

    AIR -- Yes --> AIR1[Use request airline]
    AIR -- No --> AIR2[Use ADVISOR_DEFAULT_AIRLINE\nor UNKNOWN]

    AIR1 --> DEP{departure time provided?}
    AIR2 --> DEP
    DEP -- Yes --> DEP1[Use and clamp request HHMM]
    DEP -- No --> DEP2[Average predictions\nacross 24 hours]

    DEP1 --> PRED[Run model or weekly estimator]
    DEP2 --> PRED
```

Current implementation details:

- `distance`: request -> route average -> global average
- `airline`: request -> `ADVISOR_DEFAULT_AIRLINE` -> `UNKNOWN`
- `scheduled_departure`: request -> clamped HHMM, otherwise 24-hour average
- `origin_airport` and `destination_airport` are still mandatory for delay assessment
- `country` is inferred from the airport index for UI route updates when possible, but it is not part of the same predictor fallback chain

---

## 6. LLM Provider Selection

```mermaid
flowchart LR
    ENV[ADVISOR_LLM_PROVIDER] --> NV{nvidia?}
    NV -- Yes --> NVIDIA[NVIDIA-compatible\nAPI endpoint]
    NV -- No --> HF{huggingface?}
    HF -- Yes --> HFAPI[Hugging Face\nRouter API]
    HF -- No --> FB[Fallback:\nheuristic response]

    NVIDIA & HFAPI --> PROMPT[Structured advisor\ncontext prompt]
    PROMPT --> RESP[LLM response text]
    RESP --> MERGE[Merge into\n/advise response]
```

**Token budget rules:**

| Scenario | Budget source |
|---|---|
| Compact mode (`ADVISOR_LLM_COMPACT_MODE=1`) | Reduced ceiling, shorter history |
| Qwen-like model detected | `QWEN_MAX_TOKENS` ceiling |
| Travel guide request | `ADVISOR_LLM_GUIDE_MAX_TOKENS` |
| Default | Standard advisor budget |

---

## 7. Session & Route State

```mermaid
stateDiagram-v2
    [*] --> Empty: New session or page load

    Empty --> RouteSet: User provides\nairport or country
    RouteSet --> RouteSet: User mentions\nnew route (overrides)
    RouteSet --> Predicting: Full context\npresent
    RouteSet --> WeeklyEstimate: Partial context

    Predicting --> RouteSet: Follow-up question
    WeeklyEstimate --> RouteSet: Follow-up question

    RouteSet --> Empty: POST /api/advisor/reset\nor new advisor screen
    Predicting --> Empty: POST /api/advisor/reset
    WeeklyEstimate --> Empty: POST /api/advisor/reset
```

---

## 8. Repository → Runtime Mapping

```
src/
├── app.py                        ← Deployment entrypoint
└── api/
    ├── main.py                   ← Flask factory, schemas, predictor, bootstrap
    ├── views/
    │   ├── pages.py              ← GET / /front /flight /predictions /advisor
    │   ├── flight.py             ← GET /api/flight/countries /airports /departures
    │   └── advisor.py            ← POST /advise  GET /history  POST /reset
    └── services/
        ├── llm_service.py        ← NVIDIA / HF transport, prompt assembly
        └── OpenSky.py            ← Live flight integration

models/
├── delay_model.pkl               ← Serialized ML model
├── delay_model_meta.json         ← Feature names, thresholds
└── explain/                      ← SHAP exports for top_factors

data/
└── runtime/
    └── advisor_sessions/         ← Per-session chat history (JSON files)

src/jobs/
├── generate_future_flights.py    ← Future schedule generation
├── weekly_pipeline.py            ← Weekly processing orchestration
├── weekly_predict.py             ← Weekly prediction output
└── csv_to_parquet_converter.py   ← Data format helper

dashboard/
└── app.py                        ← Optional Dash analytics (ENABLE_DASH=1)
```

---

## 9. Deployment Topology

```mermaid
flowchart LR
    subgraph client[Client]
        B[Browser]
    end

    subgraph server[Server / Railway]
        GU[gunicorn\n-w 2]
        GU --> W1[Worker 1\nsrc.app:app]
        GU --> W2[Worker 2\nsrc.app:app]
    end

    subgraph external[External Services]
        OS[OpenSky API]
        LLM[LLM Provider\nNVIDIA · Hugging Face]
    end

    subgraph fs[Filesystem]
        MD[models/]
        DD[data/runtime/]
    end

    B --> GU
    W1 & W2 --> OS
    W1 & W2 --> LLM
    W1 & W2 --> MD
    W1 & W2 --> DD
```

> ⚠️ Both workers share the same filesystem. Session files and model artifacts are read/written from disk — ensure your deployment platform provides a persistent volume for `data/runtime/` if session continuity across restarts is required.

---

## 10. Key Design Decisions

| Decision | Rationale |
|---|---|
| Three-tier prediction fallback | Never block the user — always return useful information even with partial input |
| Route context as session state | Keeps dropdowns in sync without requiring the user to re-enter route details |
| Pluggable LLM provider | Allows swapping between NVIDIA and Hugging Face without changing the advisor logic |
| Heuristic fallback when LLM is off | Advisor is functional even without LLM credentials — useful for local development |
| Weekly schedule from generated data | Decouples route estimation from live booking APIs that are not yet integrated |
