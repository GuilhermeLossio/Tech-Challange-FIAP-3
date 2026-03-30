# Flight Advisor — Architecture

> This document describes the current runtime architecture. For the live API docs see [Flight Advisor API Docs](https://flightadvisor.up.railway.app/docs/). For deployment and configuration see [`PROTOTYPE.md`](./PROTOTYPE.md).

---

## 1. System Overview

Flight Advisor is structured in four horizontal layers:

```
┌─────────────────────────────────────────────────────┐
│                  Delivery Layer                     │
│   Flask pages · Jinja2 templates · Static assets    │
├─────────────────────────────────────────────────────┤
│                    API Layer                        │
│   View registration · Request schemas · Routing     │
├─────────────────────────────────────────────────────┤
│                 Intelligence Layer                  │
│   Delay model · Weekly estimator · LLM service      │
├─────────────────────────────────────────────────────┤
│                    Data Layer                       │
│   Model artifacts · Airports index · Session store  │
│   OpenSky · Weekly schedule · SHAP exports          │
└─────────────────────────────────────────────────────┘
```

---

## 2. Runtime Boundaries

Shows system boundaries and how internal components connect to persistent data and external providers.

```mermaid
flowchart LR
    B[Browser] --> P[Flask pages + static assets]
    P --> V[HTTP routes\nsrc/api/main.py + views]

    subgraph runtime[Flight Advisor Runtime]
        V --> A[Advisor orchestration]
        V --> PR[Prediction endpoints]
        V --> LF[Live flights endpoints]
        V --> DM[Dash mount /dashboard]

        A --> N[Route extraction + context merge]
        A --> SP[Specific-flight predictor]
        A --> WR[Weekly route estimator]
        A --> LA[LLM adapter]
    end

    subgraph data[Persistent data]
        M[Model artifacts]
        AP[Airports index]
        WF[Future flights dataset]
        SS[Advisor session files]
        EX[SHAP exports]
    end

    subgraph external[External providers]
        OS[OpenSky Network]
        LP[NVIDIA or Hugging Face]
    end

    A --> SS
    A --> AP
    A --> M
    A --> WF
    A --> EX
    PR --> M
    PR --> WF
    LF --> OS
    LA --> LP
```

---

## 3. Advisor Orchestration

Shows session identity, message persistence, and the three prediction branches in their real execution order.

```mermaid
sequenceDiagram
    actor U as User
    participant UI as Advisor UI
    participant API as POST /advise
    participant SID as Flask session
    participant FS as JSON session store
    participant ORCH as Advisor orchestration
    participant PRED as Prediction engine
    participant WEEK as Weekly estimator
    participant LLM as LLM adapter

    U->>UI: Ask a question or set a route
    UI->>API: Send structured payload
    API->>SID: Get or create advisor_session_id
    API->>FS: Load message history by session_id
    API->>ORCH: Merge payload + history + extracted route hints

    alt Route incomplete
        ORCH-->>API: Discovery response + clarification prompts
    else Route complete, flight context complete
        ORCH->>PRED: Run specific-flight prediction
        PRED-->>ORCH: probability + prediction + top factors
    else Route complete, flight context partial
        ORCH->>WEEK: Run weekly route estimation
        WEEK-->>ORCH: aggregate risk + suggested flights
    end

    opt LLM enabled and question present
        ORCH->>LLM: Send structured context + history snapshot
        LLM-->>ORCH: Advice text
    end

    ORCH->>FS: Persist updated messages
    API-->>UI: advice + metrics + route_updates + messages
    UI-->>U: Render answer and sync dropdowns
```

**Session notes:**

- Session identity (`advisor_session_id`) lives in the Flask session cookie — it is browser-scoped and not persisted to disk.
- Message history is persisted separately in `data/runtime/advisor_sessions/` keyed by session ID.
- `POST /api/advisor/reset` clears both the in-memory route context and the persisted message file.
- Navigating to a new advisor screen resets route context but does not clear message history unless reset is explicitly called.

---

## 4. Delay Assessment & Input Resolution

Combines prediction mode selection and feature fallback into one flow. Fallback only applies once the system has confirmed it has enough route context to attempt a prediction.

```mermaid
flowchart TD
    Q[Incoming /advise request] --> C1{origin + destination\nknown?}

    C1 -- No --> D[Discovery mode\nreturn clarification prompts]
    C1 -- Yes --> C2{airline + departure +\nany date context known?}

    C2 -- Yes --> SF[Specific-flight prediction]
    C2 -- No --> WR[Weekly route estimation]

    subgraph resolve[Predictive input resolution]
        DIST{distance\nprovided?}
        DIST -- Yes --> D1[Use request distance]
        DIST -- No --> D2{route average\nexists?}
        D2 -- Yes --> D3[Use route-average distance]
        D2 -- No --> D4[Use global-average distance]

        AIR{airline\nprovided?}
        AIR -- Yes --> A1[Use request airline]
        AIR -- No --> A2[Use ADVISOR_DEFAULT_AIRLINE\nor UNKNOWN]

        DEP{scheduled departure\nprovided?}
        DEP -- Yes --> T1[Clamp HHMM]
        DEP -- No --> T2[Average predictions\nacross 24h]
    end

    SF --> resolve
    WR --> resolve

    D1 & D3 & D4 --> AIR
    A1 & A2 --> DEP
    T1 & T2 --> OUT[Compute risk output]

    D --> LLM{LLM enabled?}
    OUT --> LLM
    LLM -- Yes --> R1[Return advice text + metrics]
    LLM -- No --> R2[Return heuristic text + metrics]
```

**Resolution rules:**

| Feature | Resolution order |
|---|---|
| `distance` | Request → route historical average → global average |
| `airline` | Request → `ADVISOR_DEFAULT_AIRLINE` → `UNKNOWN` |
| `scheduled_departure` | Request (clamped HHMM) → average across 24-hour window |
| `origin_airport` / `destination_airport` | **Required** — no fallback; triggers discovery mode if absent |
| `origin_country` / `destination_country` | Inferred from airport index for UI `route_updates`; not part of the predictor fallback chain |

---

## 5. LLM Provider Selection

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

## 6. Deployment Topology

```mermaid
flowchart LR
    subgraph client[Client]
        BR[Browser]
    end

    subgraph ingress[HTTP runtime]
        GU[gunicorn]
        W1[Flask worker 1]
        W2[Flask worker 2]
    end

    subgraph localfs[Shared filesystem]
        MOD[models/]
        RUN[data/runtime/]
        FUT[future flights data]
    end

    subgraph ext[External services]
        OSP[OpenSky Network]
        LLM[LLM provider]
    end

    BR --> GU
    GU --> W1
    GU --> W2

    W1 & W2 --> MOD
    W1 & W2 --> RUN
    W1 & W2 --> FUT
    W1 & W2 --> OSP
    W1 & W2 --> LLM
```

> ⚠️ Both workers share the same filesystem. Session files and model artifacts are read/written from disk — ensure your deployment platform provides a persistent volume for `data/runtime/` if session continuity across restarts is required.

```bash
# Production
gunicorn -w 2 -b 0.0.0.0:$PORT src.app:app

# Plain Python on Railway
python src/app.py   # reads PORT automatically
```

---

## 7. Repository → Runtime Mapping

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

## 8. Key Design Decisions

| Decision | Rationale |
|---|---|
| Three-tier prediction fallback | Never block the user — always return useful information even with partial input |
| Route context as session state | Keeps dropdowns in sync without requiring the user to re-enter route details |
| Pluggable LLM provider | Allows swapping between NVIDIA and Hugging Face without changing the advisor logic |
| Heuristic fallback when LLM is off | Advisor is functional even without LLM credentials — useful for local development |
| Weekly schedule from generated data | Decouples route estimation from live booking APIs that are not yet integrated |
| OpenSky as external provider | Live flight data quality and availability are outside the application's control |
