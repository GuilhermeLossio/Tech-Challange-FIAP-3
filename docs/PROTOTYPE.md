# 🗂️ Flight Advisor — Project Prototyping

> Technical reference document for development. Contains the data structure, feature mapping, dashboard screen flow, and RAG architecture.

---

## Table of Contents

- [0. Diagramas de Arquitetura](#0-diagramas-de-arquitetura)
  - [0.1 Fluxo de dados entre os 3 CSVs](#01-fluxo-de-dados-entre-os-3-csvs)
  - [0.2 Pipeline de processamento](#02-pipeline-de-processamento)
  - [0.3 Arquitetura MVC completa](#03-arquitetura-mvc-completa)
  - [0.4 Fluxo de uma consulta ao Advisor](#04-fluxo-de-uma-consulta-ao-advisor)
  - [0.5 Pipeline de Treinamento (SVG)](#05-pipeline-de-treinamento-svg)
  - [0.6 Pipeline do RAG (SVG)](#06-pipeline-do-rag-svg)
- [1. Data Schema](#1-data-schema)
- [2. ML Model Feature Mapping](#2-ml-model-feature-mapping)
- [3. Dashboard Screen Flow](#3-dashboard-screen-flow)
- [4. RAG Document Structure](#4-rag-document-structure)

---

## 0. Diagramas de Arquitetura

### 0.1 Fluxo de dados entre os 3 CSVs

```mermaid
erDiagram
    FLIGHTS_CSV {
        int     YEAR
        int     MONTH
        int     DAY
        int     DAY_OF_WEEK
        string  AIRLINE           FK
        int     FLIGHT_NUMBER
        string  ORIGIN_AIRPORT    FK
        string  DESTINATION_AIRPORT FK
        int     SCHEDULED_DEPARTURE
        float   ARRIVAL_DELAY
        int     CANCELLED
        float   WEATHER_DELAY
        float   AIRLINE_DELAY
        float   AIR_SYSTEM_DELAY
        float   LATE_AIRCRAFT_DELAY
    }

    AIRPORTS_CSV {
        string  IATA_CODE         PK
        string  AIRPORT
        string  CITY
        string  STATE
        float   LATITUDE
        float   LONGITUDE
    }

    AIRLINES_CSV {
        string  IATA_CODE         PK
        string  AIRLINE
    }

    FLIGHTS_CSV }o--|| AIRLINES_CSV        : "AIRLINE → IATA_CODE"
    FLIGHTS_CSV }o--|| AIRPORTS_CSV        : "ORIGIN_AIRPORT → IATA_CODE"
    FLIGHTS_CSV }o--|| AIRPORTS_CSV        : "DESTINATION_AIRPORT → IATA_CODE"
```

### 0.2 Pipeline de processamento

```mermaid
flowchart TD
    A[(flights.csv\n~5.8M rows · 31 cols)] --> C
    B[(airports.csv\n~330 rows · 7 cols)]  --> C
    D[(airlines.csv\n~15 rows · 2 cols)]   --> C

    C[Join ORIGIN_AIRPORT + DESTINATION_AIRPORT\n+ AIRLINE via IATA_CODE]

    C --> E[Limpeza\nRemove cancelados · trata nulos\nremove outliers ARRIVAL_DELAY > 500]

    E --> F[Engenharia de features\nPERIODO_DIA · ESTACAO · IS_FERIADO\nROTA = ORIGIN_AIRPORT + DEST_AIRPORT]

    F --> G[Target encoding histórico\nORIGIN_DELAY_RATE · DEST_DELAY_RATE\nCARRIER_DELAY_RATE · ROTA_DELAY_RATE\nCARRIER_DELAY_RATE_DOW\ncalculado apenas no treino — sem data leakage]

    G --> H[Criação do target\nIS_DELAYED = ARRIVAL_DELAY > 15]

    H --> I[(flights_processed\n~27 colunas)]
    H --> J[(airport_profiles\n~18 colunas por aeroporto)]

    I --> K[Train / Test split\nstratify=IS_DELAYED]
    J --> L[RAG indexer\nFAISS vector store]
    J --> M[Clustering\nK-Means + PCA]
```

### 0.3 Arquitetura MVC completa

```mermaid
flowchart TB
    subgraph VIEW ["VIEW — Dash · dashboard/app.py"]
        V1[Overview\nKPIs · bar · line]
        V2[Explorer\nheatmap · boxplot]
        V3[Mapas geo\nscatter · route map]
        V4[Clusters\nPCA · anomalias]
        V5[Flight Advisor\nform · badge · LLM]
    end

    subgraph CONTROLLER ["CONTROLLER — src/"]
        C1[Dashboard ctrl\nfiltros · agg · charts]
        C2[Map ctrl\nlat·lon · plotly fig]
        C3[Cluster ctrl\nK-Means · PCA · IsoForest]
        C4[Advisor ctrl\nFastAPI POST /advise]
        CP[Preprocessing pipeline\nsrc/preprocessing.py]
        CS[SHAP explainer\nsrc/explainer.py]
        CR[RAG orchestrator\nsrc/rag/advisor.py]
    end

    subgraph MODEL ["MODEL — data/ · models/ · src/rag/"]
        M1[(flights.csv)]
        M2[(airports.csv)]
        M3[(airlines.csv)]
        M4[(flights_processed)]
        M5[(airport_profiles)]
        M6[(ML models\nRF · XGBoost · LogReg)]
        M7[(FAISS\nvector store)]
        M8([Qwen 3\nHF Spaces])
        M9[(AWS S3\nroadmap)]
    end

    V1 & V2 --> C1
    V3 --> C2
    V4 --> C3
    V5 --> C4

    C1 & C2 --> CP
    C3 --> CS
    C4 --> CS
    C4 --> CR

    CP --> M4
    M1 & M2 & M3 --> CP

    M4 --> M6
    M4 --> M7
    M5 --> M7
    M7 --> M8

    C1 --> M4
    C2 --> M5
    C3 --> M6
    CR --> M7
    CR --> M8

    M6 & M7 & M4 -.->|roadmap| M9
```

### 0.4 Fluxo de uma consulta ao Advisor

```mermaid
sequenceDiagram
    actor U as Usuário
    participant D as Dash\ndashboard
    participant A as FastAPI\n/advise
    participant P as Preprocessing\nsrc/preprocessing.py
    participant ML as ML Model\nXGBoost .pkl
    participant SH as SHAP Explainer\nsrc/explainer.py
    participant RV as RAG Retriever\nsrc/rag/retriever.py
    participant FS as FAISS\nvector store
    participant Q as Qwen 3\nHF Spaces

    U->>D: Preenche form\n(rota, airline, horário, pergunta)
    D->>A: POST /advise\n{origin_airport, destination_airport,\nairline, scheduled_departure,\nmonth, day_of_week, question}

    A->>P: Aplica feature engineering\n(PERIODO_DIA, ESTACAO, IS_FERIADO,\ntaxas históricas de atraso)
    P-->>A: feature_vector (13 features)

    A->>ML: predict_proba(feature_vector)
    ML-->>A: delay_probability = 0.71

    A->>SH: explain(feature_vector)
    SH-->>A: shap_values → top_factors\n[{jfk_peak_hour: +24%}, ...]

    A->>RV: retrieve(question + rota + airline)
    RV->>FS: busca exata por metadados\n(origin, dest, carrier) k=2\n+ busca semântica k=4
    FS-->>RV: docs relevantes
    RV-->>A: context (~800 tokens)

    A->>Q: prompt = system + flight_data\n+ ml_prediction + shap_factors\n+ rag_context + instructions
    Q-->>A: advice (linguagem natural)

    A-->>D: {delay_probability, risk_level,\ntop_factors, advice}
    D-->>U: Risk badge + SHAP chart\n+ resposta do Advisor
```

### 0.5 Pipeline de Treinamento (SVG)

<svg width="100%" viewBox="0 0 680 780">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- ── PHASE LABELS (left) ── -->
  <text class="ts" x="14" y="122" text-anchor="middle" transform="rotate(-90,14,122)" style="font-weight:500;fill:var(--color-text-secondary)">local</text>
  <text class="ts" x="14" y="362" text-anchor="middle" transform="rotate(-90,14,362)" style="font-weight:500;fill:var(--color-text-secondary)">aws</text>
  <text class="ts" x="14" y="622" text-anchor="middle" transform="rotate(-90,14,622)" style="font-weight:500;fill:var(--color-text-secondary)">predict</text>

  <!-- ── PHASE BANDS ── -->
  <rect x="28" y="44"  width="636" height="196" rx="10" fill="none" stroke="var(--color-border-tertiary)" stroke-width="0.5" stroke-dasharray="4 3"/>
  <rect x="28" y="260" width="636" height="220" rx="10" fill="none" stroke="var(--color-border-tertiary)" stroke-width="0.5" stroke-dasharray="4 3"/>
  <rect x="28" y="500" width="636" height="240" rx="10" fill="none" stroke="var(--color-border-tertiary)" stroke-width="0.5" stroke-dasharray="4 3"/>

  <!-- ══════════════════════
       FASE LOCAL
  ══════════════════════ -->

  <!-- flights.csv / airports.csv / airlines.csv -->
  <g class="node c-gray" onclick="sendPrompt('Quais são os 3 CSVs de entrada do projeto?')">
    <rect x="44" y="66" width="130" height="50" rx="6" stroke-width="0.5"/>
    <text class="th" x="109" y="84" text-anchor="middle" dominant-baseline="central">Dados brutos</text>
    <text class="ts" x="109" y="102" text-anchor="middle" dominant-baseline="central">flights · airports · airlines</text>
  </g>

  <!-- Preprocessing -->
  <g class="node c-purple" onclick="sendPrompt('O que o pipeline de preprocessing faz?')">
    <rect x="228" y="66" width="150" height="50" rx="6" stroke-width="0.5"/>
    <text class="th" x="303" y="84" text-anchor="middle" dominant-baseline="central">Preprocessing</text>
    <text class="ts" x="303" y="102" text-anchor="middle" dominant-baseline="central">join · limpeza · features</text>
  </g>

  <!-- Treinamento local -->
  <g class="node c-purple" onclick="sendPrompt('Quais modelos são treinados localmente?')">
    <rect x="440" y="66" width="150" height="50" rx="6" stroke-width="0.5"/>
    <text class="th" x="515" y="84" text-anchor="middle" dominant-baseline="central">Treinamento local</text>
    <text class="ts" x="515" y="102" text-anchor="middle" dominant-baseline="central">RF · XGBoost · LogReg</text>
  </g>

  <!-- SHAP -->
  <g class="node c-amber" onclick="sendPrompt('Como o SHAP é usado no treinamento?')">
    <rect x="440" y="170" width="150" height="44" rx="6" stroke-width="0.5"/>
    <text class="th" x="515" y="189" text-anchor="middle" dominant-baseline="central">SHAP explainer</text>
    <text class="ts" x="515" y="205" text-anchor="middle" dominant-baseline="central">src/explainer.py</text>
  </g>

  <!-- airport_profiles -->
  <g class="node c-gray" onclick="sendPrompt('Como airport_profiles é gerado?')">
    <rect x="228" y="170" width="150" height="44" rx="6" stroke-width="0.5"/>
    <text class="th" x="303" y="189" text-anchor="middle" dominant-baseline="central">airport_profiles</text>
    <text class="ts" x="303" y="205" text-anchor="middle" dominant-baseline="central">18 cols · clustering</text>
  </g>

  <!-- arrows fase local -->
  <line x1="174" y1="91" x2="226" y2="91" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <line x1="378" y1="91" x2="438" y2="91" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <line x1="515" y1="116" x2="515" y2="168" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <line x1="303" y1="116" x2="303" y2="168" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>

  <!-- ══════════════════════
       FASE AWS
  ══════════════════════ -->

  <!-- S3 raw -->
  <g class="node c-teal" onclick="sendPrompt('O que é enviado para o S3 após o treinamento local?')">
    <rect x="44" y="282" width="140" height="50" rx="6" stroke-width="0.5"/>
    <text class="th" x="114" y="300" text-anchor="middle" dominant-baseline="central">AWS S3</text>
    <text class="ts" x="114" y="318" text-anchor="middle" dominant-baseline="central">modelos · dados raw</text>
  </g>

  <!-- Refinamento -->
  <g class="node c-teal" onclick="sendPrompt('Como funciona o refinamento do modelo na nuvem?')">
    <rect x="252" y="282" width="150" height="50" rx="6" stroke-width="0.5"/>
    <text class="th" x="327" y="300" text-anchor="middle" dominant-baseline="central">Refinamento</text>
    <text class="ts" x="327" y="318" text-anchor="middle" dominant-baseline="central">fine-tune · validação</text>
  </g>

  <!-- Athena -->
  <g class="node c-teal" onclick="sendPrompt('Qual é o papel do Athena no projeto?')">
    <rect x="468" y="282" width="168" height="50" rx="6" stroke-width="0.5"/>
    <text class="th" x="552" y="300" text-anchor="middle" dominant-baseline="central">AWS Athena</text>
    <text class="ts" x="552" y="318" text-anchor="middle" dominant-baseline="central">database de resultados</text>
  </g>

  <!-- modelo refinado pkl -->
  <g class="node c-teal" onclick="sendPrompt('O modelo refinado fica serializado no S3 ou no Athena?')">
    <rect x="252" y="394" width="150" height="44" rx="6" stroke-width="0.5"/>
    <text class="th" x="327" y="413" text-anchor="middle" dominant-baseline="central">Modelo refinado</text>
    <text class="ts" x="327" y="429" text-anchor="middle" dominant-baseline="central">.pkl salvo no S3</text>
  </g>

  <!-- arrows fase aws -->
  <!-- treinamento local → S3 (cruzando fase) -->
  <path d="M515 240 L515 252 L114 252 L114 280" fill="none" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <line x1="184" y1="307" x2="250" y2="307" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <line x1="402" y1="307" x2="466" y2="307" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <line x1="327" y1="332" x2="327" y2="392" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <!-- athena → modelo refinado (resultado do refinamento vai pro athena) -->
  <path d="M552 332 L552 416 L404 416" fill="none" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>

  <!-- ══════════════════════
       FASE PREDICT
  ══════════════════════ -->

  <!-- Job de predição -->
  <g class="node c-blue" onclick="sendPrompt('Como funciona o job de predição semanal?')">
    <rect x="44" y="526" width="168" height="50" rx="6" stroke-width="0.5"/>
    <text class="th" x="128" y="544" text-anchor="middle" dominant-baseline="central">Job de predição</text>
    <text class="ts" x="128" y="562" text-anchor="middle" dominant-baseline="central">lê Athena · roda modelo</text>
  </g>

  <!-- Previsão semana seguinte -->
  <g class="node c-blue" onclick="sendPrompt('O que a previsão da semana seguinte retorna?')">
    <rect x="272" y="526" width="168" height="50" rx="6" stroke-width="0.5"/>
    <text class="th" x="356" y="544" text-anchor="middle" dominant-baseline="central">Previsão semanal</text>
    <text class="ts" x="356" y="562" text-anchor="middle" dominant-baseline="central">próximos 7 dias de voos</text>
  </g>

  <!-- S3 Predict -->
  <g class="node c-blue" onclick="sendPrompt('O que vai para o S3 como Predict?')">
    <rect x="500" y="526" width="148" height="50" rx="6" stroke-width="0.5"/>
    <text class="th" x="574" y="544" text-anchor="middle" dominant-baseline="central">AWS S3</text>
    <text class="ts" x="574" y="562" text-anchor="middle" dominant-baseline="central">resultados como "Predict"</text>
  </g>

  <!-- Dash / API consome -->
  <g class="node c-gray" onclick="sendPrompt('Como o Dash e a API consomem as previsões do S3?')">
    <rect x="272" y="648" width="168" height="50" rx="6" stroke-width="0.5"/>
    <text class="th" x="356" y="666" text-anchor="middle" dominant-baseline="central">Dash + FastAPI</text>
    <text class="ts" x="356" y="684" text-anchor="middle" dominant-baseline="central">exibe previsões ao usuário</text>
  </g>

  <!-- arrows fase predict -->
  <!-- modelo refinado → job -->
  <path d="M327 440 L327 490 L128 490 L128 524" fill="none" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <!-- athena → job -->
  <path d="M552 446 L552 490 L128 490" fill="none" stroke="var(--color-border-secondary)" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#arrow)"/>
  <line x1="212" y1="551" x2="270" y2="551" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <line x1="440" y1="551" x2="498" y2="551" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <line x1="356" y1="576" x2="356" y2="646" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <!-- S3 predict → dash -->
  <path d="M574 576 L574 672 L442 672" fill="none" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>

  <!-- label linha tracejada Athena → job -->
  <text class="ts" x="340" y="485" text-anchor="middle" style="fill:var(--color-text-secondary)">dados históricos</text>

  <!-- ── LEGEND ── -->
  <rect x="44"  y="748" width="12" height="12" rx="2" fill="#D3D1C7"/>
  <text class="ts" x="62"  y="759">Dados / storage local</text>
  <rect x="180" y="748" width="12" height="12" rx="2" fill="#CECBF6"/>
  <text class="ts" x="198" y="759">Pipeline ML local</text>
  <rect x="316" y="748" width="12" height="12" rx="2" fill="#9FE1CB"/>
  <text class="ts" x="334" y="759">AWS (S3 · Athena)</text>
  <rect x="452" y="748" width="12" height="12" rx="2" fill="#B5D4F4"/>
  <text class="ts" x="470" y="759">Predict pipeline</text>
</svg>

### 0.6 Pipeline do RAG (SVG)

<svg width="100%" viewBox="0 0 680 820">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- ── PHASE LABELS ── -->
  <text class="ts" x="14" y="118" text-anchor="middle" transform="rotate(-90,14,118)" style="font-weight:500;fill:var(--color-text-secondary)">cliente</text>
  <text class="ts" x="14" y="320" text-anchor="middle" transform="rotate(-90,14,320)" style="font-weight:500;fill:var(--color-text-secondary)">rag</text>
  <text class="ts" x="14" y="548" text-anchor="middle" transform="rotate(-90,14,548)" style="font-weight:500;fill:var(--color-text-secondary)">llm</text>
  <text class="ts" x="14" y="714" text-anchor="middle" transform="rotate(-90,14,714)" style="font-weight:500;fill:var(--color-text-secondary)">dados</text>

  <!-- ── PHASE BANDS ── -->
  <rect x="28" y="44"  width="636" height="168" rx="10" fill="none" stroke="var(--color-border-tertiary)" stroke-width="0.5" stroke-dasharray="4 3"/>
  <rect x="28" y="232" width="636" height="196" rx="10" fill="none" stroke="var(--color-border-tertiary)" stroke-width="0.5" stroke-dasharray="4 3"/>
  <rect x="28" y="448" width="636" height="196" rx="10" fill="none" stroke="var(--color-border-tertiary)" stroke-width="0.5" stroke-dasharray="4 3"/>
  <rect x="28" y="664" width="636" height="120" rx="10" fill="none" stroke="var(--color-border-tertiary)" stroke-width="0.5" stroke-dasharray="4 3"/>

  <!-- ══════════════════════
       FASE CLIENTE
  ══════════════════════ -->

  <!-- Dash — formulário -->
  <g class="node c-blue" onclick="sendPrompt('Como funciona o formulário de voo no Dash para o RAG?')">
    <rect x="44" y="66" width="168" height="56" rx="6" stroke-width="0.5"/>
    <text class="th" x="128" y="87" text-anchor="middle" dominant-baseline="central">Dash — formulário</text>
    <text class="ts" x="128" y="107" text-anchor="middle" dominant-baseline="central">origem · destino · data</text>
  </g>

  <!-- Dash — chat -->
  <g class="node c-blue" onclick="sendPrompt('Como funciona o chat livre no Dash para o RAG?')">
    <rect x="256" y="66" width="168" height="56" rx="6" stroke-width="0.5"/>
    <text class="th" x="340" y="87" text-anchor="middle" dominant-baseline="central">Dash — chat</text>
    <text class="ts" x="340" y="107" text-anchor="middle" dominant-baseline="central">linguagem natural livre</text>
  </g>

  <!-- FastAPI /rag-advise -->
  <g class="node c-blue" onclick="sendPrompt('O que o endpoint /rag-advise recebe e retorna?')">
    <rect x="468" y="66" width="168" height="56" rx="6" stroke-width="0.5"/>
    <text class="th" x="552" y="87" text-anchor="middle" dominant-baseline="central">FastAPI</text>
    <text class="ts" x="552" y="107" text-anchor="middle" dominant-baseline="central">POST /rag-advise</text>
  </g>

  <!-- seta formulário → FastAPI -->
  <path d="M212 94 L256 94" fill="none" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <!-- seta chat → FastAPI -->
  <path d="M424 94 L466 94" fill="none" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>

  <!-- label abaixo dos inputs -->
  <text class="ts" x="340" y="158" text-anchor="middle" style="fill:var(--color-text-secondary)">query estruturada + pergunta livre</text>
  <line x1="552" y1="122" x2="552" y2="170" fill="none" stroke="var(--color-border-secondary)" stroke-width="1.5" marker-end="url(#arrow)"/>
  <path d="M340 122 L340 158 L552 158 L552 170" fill="none" stroke="var(--color-border-secondary)" stroke-width="1" stroke-dasharray="4 3"/>

  <!-- ══════════════════════
       FASE RAG
  ══════════════════════ -->

  <!-- RAG Retriever -->
  <g class="node c-purple" onclick="sendPrompt('Como o RAG retriever funciona com o Athena?')">
    <rect x="44" y="254" width="168" height="56" rx="6" stroke-width="0.5"/>
    <text class="th" x="128" y="275" text-anchor="middle" dominant-baseline="central">RAG retriever</text>
    <text class="ts" x="128" y="295" text-anchor="middle" dominant-baseline="central">src/rag/retriever.py</text>
  </g>

  <!-- FAISS vector store -->
  <g class="node c-purple" onclick="sendPrompt('Como o FAISS é indexado a partir do Athena?')">
    <rect x="256" y="254" width="168" height="56" rx="6" stroke-width="0.5"/>
    <text class="th" x="340" y="275" text-anchor="middle" dominant-baseline="central">FAISS vector store</text>
    <text class="ts" x="340" y="295" text-anchor="middle" dominant-baseline="central">indexado do Athena</text>
  </g>

  <!-- Contexto montado -->
  <g class="node c-purple" onclick="sendPrompt('O que entra no contexto final enviado ao Qwen 3?')">
    <rect x="468" y="254" width="168" height="56" rx="6" stroke-width="0.5"/>
    <text class="th" x="552" y="275" text-anchor="middle" dominant-baseline="central">Contexto montado</text>
    <text class="ts" x="552" y="295" text-anchor="middle" dominant-baseline="central">previsões · rotas · padrões</text>
  </g>

  <!-- RAG indexer (abaixo) -->
  <g class="node c-purple" onclick="sendPrompt('Como funciona o indexer que popula o FAISS?')">
    <rect x="256" y="358" width="168" height="44" rx="6" stroke-width="0.5"/>
    <text class="th" x="340" y="377" text-anchor="middle" dominant-baseline="central">RAG indexer</text>
    <text class="ts" x="340" y="393" text-anchor="middle" dominant-baseline="central">src/rag/indexer.py</text>
  </g>

  <!-- setas fase RAG -->
  <path d="M552 170 L552 232 L128 232 L128 252" fill="none" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <line x1="212" y1="282" x2="254" y2="282" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <line x1="424" y1="282" x2="466" y2="282" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <line x1="340" y1="310" x2="340" y2="356" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>

  <!-- ══════════════════════
       FASE LLM
  ══════════════════════ -->

  <!-- Prompt builder -->
  <g class="node c-teal" onclick="sendPrompt('O que o prompt builder monta antes de chamar o Qwen 3?')">
    <rect x="44" y="470" width="168" height="56" rx="6" stroke-width="0.5"/>
    <text class="th" x="128" y="491" text-anchor="middle" dominant-baseline="central">Prompt builder</text>
    <text class="ts" x="128" y="511" text-anchor="middle" dominant-baseline="central">system · query · contexto</text>
  </g>

  <!-- Qwen 3 HF API -->
  <g class="node c-teal" onclick="sendPrompt('Como o Qwen 3 é chamado via API no Hugging Face?')">
    <rect x="256" y="470" width="168" height="56" rx="6" stroke-width="0.5"/>
    <text class="th" x="340" y="491" text-anchor="middle" dominant-baseline="central">Qwen 3</text>
    <text class="ts" x="340" y="511" text-anchor="middle" dominant-baseline="central">HF Spaces API</text>
  </g>

  <!-- Resposta -->
  <g class="node c-teal" onclick="sendPrompt('O que o Qwen 3 retorna para o usuário?')">
    <rect x="468" y="470" width="168" height="56" rx="6" stroke-width="0.5"/>
    <text class="th" x="552" y="491" text-anchor="middle" dominant-baseline="central">Resposta gerada</text>
    <text class="ts" x="552" y="511" text-anchor="middle" dominant-baseline="central">sugestão · risco · motivo</text>
  </g>

  <!-- Dash resultado -->
  <g class="node c-blue" onclick="sendPrompt('Como o Dash exibe a sugestão do Qwen 3 ao cliente?')">
    <rect x="468" y="578" width="168" height="44" rx="6" stroke-width="0.5"/>
    <text class="th" x="552" y="597" text-anchor="middle" dominant-baseline="central">Dash — resultado</text>
    <text class="ts" x="552" y="613" text-anchor="middle" dominant-baseline="central">badge · card · resposta LLM</text>
  </g>

  <!-- setas fase LLM -->
  <path d="M552 310 L552 438 L128 438 L128 468" fill="none" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <line x1="212" y1="498" x2="254" y2="498" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <line x1="424" y1="498" x2="466" y2="498" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>
  <line x1="552" y1="526" x2="552" y2="576" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>

  <!-- ══════════════════════
       FASE DADOS
  ══════════════════════ -->

  <!-- Athena -->
  <g class="node c-amber" onclick="sendPrompt('Quais dados o Athena disponibiliza para o RAG?')">
    <rect x="44" y="686" width="168" height="56" rx="6" stroke-width="0.5"/>
    <text class="th" x="128" y="707" text-anchor="middle" dominant-baseline="central">AWS Athena</text>
    <text class="ts" x="128" y="727" text-anchor="middle" dominant-baseline="central">previsões processadas</text>
  </g>

  <!-- S3 Predict -->
  <g class="node c-amber" onclick="sendPrompt('Como o S3 Predict alimenta o Athena?')">
    <rect x="256" y="686" width="168" height="56" rx="6" stroke-width="0.5"/>
    <text class="th" x="340" y="707" text-anchor="middle" dominant-baseline="central">S3 Predict</text>
    <text class="ts" x="340" y="727" text-anchor="middle" dominant-baseline="central">job semanal de previsão</text>
  </g>

  <!-- setas fase dados -->
  <!-- Athena → RAG indexer -->
  <path d="M128 684 L128 620 L340 620 L340 404" fill="none" class="arr" marker-end="url(#arrow)" stroke="#BA7517" stroke-width="1.5" stroke-dasharray="5 3"/>
  <!-- S3 Predict → Athena -->
  <line x1="256" y1="714" x2="214" y2="714" class="arr" marker-end="url(#arrow)" stroke="var(--color-border-secondary)"/>

  <!-- label linha tracejada -->
  <text class="ts" x="234" y="615" text-anchor="middle" style="fill:var(--color-text-secondary)">indexa docs para FAISS</text>

  <!-- ── LEGEND ── -->
  <rect x="44"  y="768" width="12" height="12" rx="2" fill="#B5D4F4"/>
  <text class="ts" x="62"  y="779">Dash · FastAPI (cliente)</text>
  <rect x="220" y="768" width="12" height="12" rx="2" fill="#CECBF6"/>
  <text class="ts" x="238" y="779">RAG pipeline</text>
  <rect x="340" y="768" width="12" height="12" rx="2" fill="#9FE1CB"/>
  <text class="ts" x="358" y="779">Qwen 3 · LLM</text>
  <rect x="448" y="768" width="12" height="12" rx="2" fill="#FAC775"/>
  <text class="ts" x="466" y="779">Dados AWS</text>
</svg>

---

## 1. Data Schema

The project uses **three CSV files** as data sources. `flights.csv` is the main training base; `airports.csv` and `airlines.csv` are reference tables that enrich the data via join.

```
flights.csv  ──── AIRLINE             ────►  airlines.csv (IATA_CODE)
             ──── ORIGIN_AIRPORT      ────►  airports.csv (IATA_CODE)
             ──── DESTINATION_AIRPORT ────►  airports.csv (IATA_CODE)
```

---

### 1.1 Flights Dataset — `flights.csv`

Main training source. Each row represents a domestic flight in the USA.

| # | Column | Type | Example | Description |
|---|---|---|---|---|
| 1 | `YEAR` | int | `2015` | Year of the flight |
| 2 | `MONTH` | int | `11` | Month (1–12) |
| 3 | `DAY` | int | `15` | Day of the month |
| 4 | `DAY_OF_WEEK` | int | `5` | Day of the week (1=Mon … 7=Sun) |
| 5 | `AIRLINE` | str | `DL` | IATA code of the airline — **FK → `airlines.csv`** |
| 6 | `FLIGHT_NUMBER` | int | `1234` | Flight number |
| 7 | `TAIL_NUMBER` | str | `N12345` | Aircraft identification |
| 8 | `ORIGIN_AIRPORT` | str | `JFK` | IATA code origin — **FK → `airports.csv`** |
| 9 | `DESTINATION_AIRPORT` | str | `LAX` | IATA code destination — **FK → `airports.csv`** |
| 10 | `SCHEDULED_DEPARTURE` | int | `1800` | Scheduled departure time (HHMM) |
| 11 | `DEPARTURE_TIME` | float | `1823.0` | Actual departure time (HHMM) |
| 12 | `DEPARTURE_DELAY` | float | `23.0` | Departure delay (min); negative = early |
| 13 | `TAXI_OUT` | float | `15.0` | Taxi time to the runway (min) |
| 14 | `WHEELS_OFF` | float | `1838.0` | Takeoff time (HHMM) |
| 15 | `SCHEDULED_TIME` | float | `360.0` | Scheduled flight time (min) |
| 16 | `ELAPSED_TIME` | float | `371.0` | Actual flight time (min) |
| 17 | `AIR_TIME` | float | `348.0` | Effective time in the air (min) |
| 18 | `DISTANCE` | float | `2475.0` | Route distance (miles) |
| 19 | `WHEELS_ON` | float | `2046.0` | Landing time (HHMM) |
| 20 | `TAXI_IN` | float | `8.0` | Taxi time to the gate (min) |
| 21 | `SCHEDULED_ARRIVAL` | int | `2100` | Scheduled arrival time (HHMM) |
| 22 | `ARRIVAL_TIME` | float | `2134.0` | Actual arrival time (HHMM) |
| 23 | `ARRIVAL_DELAY` | float | `34.0` | Arrival delay (min) — ⚠️ **target variable** |
| 24 | `DIVERTED` | int | `0` | 1 = flight diverted to another airport |
| 25 | `CANCELLED` | int | `0` | 1 = flight cancelled |
| 26 | `CANCELLATION_REASON` | str | `B` | A=Carrier · B=Weather · C=NAS · D=Security |
| 27 | `AIR_SYSTEM_DELAY` | float | `19.0` | Delay by air traffic control (min) |
| 28 | `SECURITY_DELAY` | float | `0.0` | Delay by security (min) |
| 29 | `AIRLINE_DELAY` | float | `15.0` | Delay by airline operational problem (min) |
| 30 | `LATE_AIRCRAFT_DELAY` | float | `0.0` | Delay caused by a delayed aircraft in the previous leg (min) |
| 31 | `WEATHER_DELAY` | float | `0.0` | Delay by weather conditions (min) |

> ⚠️ **Expected nulls:** cause columns (`AIR_SYSTEM_DELAY` … `WEATHER_DELAY`) are null when the flight was not delayed or was cancelled. `CANCELLATION_REASON` is null for non-cancelled flights. `DEPARTURE_TIME`, `ARRIVAL_TIME` and derivatives are null for cancelled flights. Handle before modeling.

---

### 1.2 Airports Dataset — `airports.csv`

Geographic reference table. Join with `flights.csv` via `IATA_CODE = ORIGIN_AIRPORT` or `DESTINATION_AIRPORT`.

| # | Column | Type | Example | Description |
|---|---|---|---|---|
| 1 | `IATA_CODE` | str | `JFK` | IATA code — **PK** |
| 2 | `AIRPORT` | str | `John F Kennedy Intl` | Full name of the airport |
| 3 | `CITY` | str | `New York` | City of the airport |
| 4 | `STATE` | str | `NY` | State (abbreviation) |
| 5 | `COUNTRY` | str | `USA` | Country |
| 6 | `LATITUDE` | float | `40.6398` | Latitude — used in maps and geographic clustering |
| 7 | `LONGITUDE` | float | `-73.7789` | Longitude — used in maps and geographic clustering |

---

### 1.3 Airlines Dataset — `airlines.csv`

Airline reference table. Join with `flights.csv` via `IATA_CODE = AIRLINE`.

| # | Column | Type | Example | Description |
|---|---|---|---|---|
| 1 | `IATA_CODE` | str | `DL` | IATA code — **PK** |
| 2 | `AIRLINE` | str | `Delta Air Lines Inc.` | Full name of the airline |

---

### 1.4 Processed Dataset — `flights_processed`

Result of the cleaning pipeline + joins + feature engineering. Feeds the ML models.

| # | Column | Type | Source | Transformation |
|---|---|---|---|---|
| 1 | `IS_DELAYED` | int | `ARRIVAL_DELAY` | `1` if `ARRIVAL_DELAY > 15`, else `0` — **target** |
| 2 | `ARRIVAL_DELAY_CLEAN` | float | `ARRIVAL_DELAY` | Nulls from cancelled → `NaN`; outliers > 500 min removed |
| 3 | `MONTH` | int | `flights.csv` | Kept |
| 4 | `DAY_OF_WEEK` | int | `flights.csv` | Kept |
| 5 | `SCHEDULED_DEPARTURE` | int | `flights.csv` | Kept |
| 6 | `DISTANCE` | float | `flights.csv` | Kept |
| 7 | `AIRLINE` | str | `flights.csv` | Kept — join key |
| 8 | `ORIGIN_AIRPORT` | str | `flights.csv` | Kept — join key |
| 9 | `DESTINATION_AIRPORT` | str | `flights.csv` | Kept — join key |
| 10 | `AIRLINE_NAME` | str | `airlines.csv` | Full name via `AIRLINE = IATA_CODE` |
| 11 | `ORIGIN_CITY` | str | `airports.csv` | Origin city via `ORIGIN_AIRPORT = IATA_CODE` |
| 12 | `ORIGIN_STATE` | str | `airports.csv` | Origin state |
| 13 | `ORIGIN_LAT` | float | `airports.csv` | Origin latitude (maps) |
| 14 | `ORIGIN_LON` | float | `airports.csv` | Origin longitude (maps) |
| 15 | `DEST_CITY` | str | `airports.csv` | Destination city via `DESTINATION_AIRPORT = IATA_CODE` |
| 16 | `DEST_STATE` | str | `airports.csv` | Destination state |
| 17 | `DEST_LAT` | float | `airports.csv` | Destination latitude (maps) |
| 18 | `DEST_LON` | float | `airports.csv` | Destination longitude (maps) |
| 19 | `PERIODO_DIA` | str | `SCHEDULED_DEPARTURE` | Derived feature — see section 2.2 |
| 20 | `ESTACAO` | str | `MONTH` | Derived feature — see section 2.2 |
| 21 | `ROTA` | str | `ORIGIN_AIRPORT` + `DESTINATION_AIRPORT` | Concatenation: `"JFK_LAX"` |
| 22 | `IS_FERIADO` | int | `YEAR`, `MONTH`, `DAY` | `1` if date is a US federal holiday (lib `holidays`) |
| 23 | `ORIGIN_DELAY_RATE` | float | historical `flights.csv` | Historical delay rate of the origin airport |
| 24 | `DEST_DELAY_RATE` | float | historical `flights.csv` | Historical delay rate of the destination airport |
| 25 | `CARRIER_DELAY_RATE` | float | historical `flights.csv` | Historical delay rate of the airline |
| 26 | `ROTA_DELAY_RATE` | float | historical `flights.csv` | Historical delay rate of the specific route |
| 27 | `CARRIER_DELAY_RATE_DOW` | float | historical `flights.csv` | Airline's delay rate on that day of the week |

---

### 1.5 Airport Profiles — `airport_profiles`

Generated in the EDA by enriching `airports.csv` with statistics from `flights.csv`. Used for clustering and RAG documents.

| # | Column | Type | Source | Description |
|---|---|---|---|---|
| 1 | `IATA_CODE` | str | `airports.csv` | IATA code — PK |
| 2 | `AIRPORT` | str | `airports.csv` | Full name |
| 3 | `CITY` | str | `airports.csv` | City |
| 4 | `STATE` | str | `airports.csv` | State |
| 5 | `LATITUDE` | float | `airports.csv` | Latitude |
| 6 | `LONGITUDE` | float | `airports.csv` | Longitude |
| 7 | `TOTAL_FLIGHTS` | int | `flights.csv` | Total flights in the period |
| 8 | `DELAY_RATE` | float | `flights.csv` | % of flights with `ARRIVAL_DELAY > 15` |
| 9 | `AVG_DELAY` | float | `flights.csv` | Average delay in minutes |
| 10 | `AVG_AIRLINE_DELAY` | float | `flights.csv` | Average `AIRLINE_DELAY` |
| 11 | `AVG_WEATHER_DELAY` | float | `flights.csv` | Average `WEATHER_DELAY` |
| 12 | `AVG_AIR_SYSTEM_DELAY` | float | `flights.csv` | Average `AIR_SYSTEM_DELAY` |
| 13 | `AVG_LATE_AIRCRAFT` | float | `flights.csv` | Average `LATE_AIRCRAFT_DELAY` |
| 14 | `WORST_MONTH` | int | `flights.csv` | Month with the highest delay rate |
| 15 | `BEST_MONTH` | int | `flights.csv` | Month with the lowest delay rate |
| 16 | `WORST_DOW` | int | `flights.csv` | Day of the week with the highest delay rate |
| 17 | `BEST_DOW` | int | `flights.csv` | Day of the week with the lowest delay rate |
| 18 | `CLUSTER` | int | K-Means | Group assigned by the clustering model |

---

## 2. ML Model Feature Mapping

### 2.1 Final Model Features

All features refer to columns in `flights_processed`. Names are aligned with the actual schema of the files.

| # | Feature | Type | Encoding | Expected Importance | Justification |
|---|---|---|---|---|---|
| 1 | `MONTH` | int | Ordinal | High | Strong seasonality: December and July are critical |
| 2 | `DAY_OF_WEEK` | int | Ordinal | High | Fridays and Sundays have a higher delay rate |
| 3 | `SCHEDULED_DEPARTURE` | int | Ordinal | High | Afternoon peak flights accumulate delays from the day |
| 4 | `DISTANCE` | float | Standard | Medium | Long flights have a lower relative delay rate |
| 5 | `PERIODO_DIA` | str | One-Hot | High | Captures the peak effect non-linearly |
| 6 | `ESTACAO` | str | One-Hot | Medium | Winter and summer have distinct behaviors |
| 7 | `IS_FERIADO` | int | Binary | Medium | Holidays increase congestion at hubs |
| 8 | `ORIGIN_DELAY_RATE` | float | Standard | Very High | Airports with a poor history tend to repeat it |
| 9 | `DEST_DELAY_RATE` | float | Standard | High | Congested destinations cause slot waiting |
| 10 | `CARRIER_DELAY_RATE` | float | Standard | Very High | Airlines with a poor history tend to repeat it |
| 11 | `ROTA_DELAY_RATE` | float | Standard | Very High | Specific routes have structural delay patterns |
| 12 | `CARRIER_DELAY_RATE_DOW` | float | Standard | High | Interaction between airline and day of the week |
| 13 | `AIRLINE` | str | One-Hot | High | Operational differences between airlines |

> ⚠️ **Columns excluded from the model (data leakage):** `DEPARTURE_TIME`, `ARRIVAL_TIME`, `WHEELS_OFF`, `WHEELS_ON`, `TAXI_OUT`, `TAXI_IN`, `ELAPSED_TIME`, `AIR_TIME` — are only known after the flight. `DEPARTURE_DELAY` has a very high correlation with the target but is also only available in real-time.

---

### 2.2 Feature Engineering Rules

| Derived Feature | Logic | Possible Values |
|---|---|---|
| `PERIODO_DIA` | `SCHEDULED_DEPARTURE < 600` → early_morning | `early_morning`, `morning`, `afternoon`, `night` |
| | `600 ≤ SCHEDULED_DEPARTURE < 1200` → morning | |
| | `1200 ≤ SCHEDULED_DEPARTURE < 1800` → afternoon | |
| | `SCHEDULED_DEPARTURE ≥ 1800` → night | |
| `ESTACAO` | `MONTH in [12, 1, 2]` → winter | `winter`, `spring`, `summer`, `autumn` |
| | `MONTH in [3, 4, 5]` → spring | |
| | `MONTH in [6, 7, 8]` → summer | |
| | `MONTH in [9, 10, 11]` → autumn | |
| `IS_FERIADO` | `holidays.US(years=YEAR)[date(YEAR, MONTH, DAY)]` | `0` or `1` |
| `IS_DELAYED` | `ARRIVAL_DELAY > 15` | `0` or `1` |
| `ROTA` | `f"{ORIGIN_AIRPORT}_{DESTINATION_AIRPORT}"` | Ex: `"JFK_LAX"` |
| `ORIGIN_DELAY_RATE` | `df.groupby("ORIGIN_AIRPORT")["IS_DELAYED"].mean()` | Float `[0, 1]` |
| `DEST_DELAY_RATE` | `df.groupby("DESTINATION_AIRPORT")["IS_DELAYED"].mean()` | Float `[0, 1]` |
| `CARRIER_DELAY_RATE` | `df.groupby("AIRLINE")["IS_DELAYED"].mean()` | Float `[0, 1]` |
| `ROTA_DELAY_RATE` | `df.groupby("ROTA")["IS_DELAYED"].mean()` | Float `[0, 1]` |
| `CARRIER_DELAY_RATE_DOW` | `df.groupby(["AIRLINE","DAY_OF_WEEK"])["IS_DELAYED"].mean()` | Float `[0, 1]` |

> ⚠️ **Avoid data leakage in historical rates:** calculate rates only on the training set and apply them to the test set via `.map()`. Never calculate on the complete dataset before the split.

---

### 2.3 Model Comparison

| Criterion | Logistic Regression | Random Forest | XGBoost |
|---|---|---|---|
| **Interpretability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Expected Performance**| ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Training Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Null Tolerance** | ❌ | ✅ | ✅ |
| **Imbalanced Classes** | `class_weight="balanced"` | `class_weight="balanced"` | `scale_pos_weight` |
| **Key Hyperparameters** | `C`, `max_iter` | `n_estimators`, `max_depth` | `learning_rate`, `n_estimators`, `max_depth` |
| **SHAP Type** | Linear SHAP | Tree SHAP | Tree SHAP |
| **Role in the project** | Baseline | Robust model | Main model |

---

### 2.4 Evaluation Metrics

| Metric | Definition | Why use it | Goal |
|---|---|---|---|
| **ROC-AUC** | Area under the ROC curve | Robust to imbalance; evaluates ranking | > 0.78 |
| **F1-Score** | `2 × (Precision × Recall) / (Precision + Recall)` | Balance for minority class | > 0.70 |
| **Precision** | `TP / (TP + FP)` | Avoid false alarms of delay | > 0.65 |
| **Recall** | `TP / (TP + FN)` | Do not miss real delays | > 0.72 |
| **Accuracy** | `(TP + TN) / Total` | General reference — misleading with imbalance | Monitor |

---

## 3. Dashboard Screen Flow

### 3.1 Screen Map

```
┌──────────┐     ┌──────────────┐     ┌──────────────────┐
│  Home /  │────►│   Explorer   │────►│  Flight Advisor  │
│ Overview │     │   (filters)  │     │   (RAG query)    │
└──────────┘     └──────────────┘     └──────────────────┘
     │                  │
     ▼                  ▼
┌──────────┐     ┌──────────────┐
│   Geo    │     │  Clusters &  │
│   Maps   │     │   Anomalies  │
└──────────┘     └──────────────┘
```

---

### 3.2 Screen 1 — Home / Overview

| Element | Type | Position | Content | Data Source |
|---|---|---|---|---|
| Header | Text | Top | "✈️ Flight Advisor — USA" + subtitle | Static |
| KPI 1 | Metric | Col 1/4 | Total flights analyzed | `len(df)` |
| KPI 2 | Metric | Col 2/4 | % of delayed flights | `IS_DELAYED.mean()` |
| KPI 3 | Metric | Col 3/4 | Average delay (min) | `ARRIVAL_DELAY.mean()` |
| KPI 4 | Metric | Col 4/4 | Airline with the highest delay | `groupby AIRLINE` + join `AIRLINE_NAME` from `airlines.csv` |
| Chart 1 | Bar chart | Row 2, left | Average delay by airline (full name) | `groupby AIRLINE` + `airlines.csv` |
| Chart 2 | Line chart | Row 2, right | Monthly evolution of delay | `groupby MONTH` |
| Sidebar | Filters | Side | Airline (`AIRLINE_NAME`), Month, Origin State (`STATE`) | `st.multiselect`, `st.slider` |

---

### 3.3 Screen 2 — Explorer (Detailed Analysis)

| Element | Type | Position | Content | Data Source |
|---|---|---|---|---|
| Route filter | Selectbox | Sidebar | Origin + Destination with city name | `airports.csv` `CITY` + `IATA_CODE` |
| Period filter | Slider | Sidebar | Start / end month | `MONTH` |
| Chart 1 | Heatmap | Row 1 | Average delay: Hour × Day of the week | `pivot_table SCHEDULED_DEPARTURE × DAY_OF_WEEK` |
| Chart 2 | Bar chart | Row 2, left | Top 10 routes with the highest delay | `groupby ROTA` |
| Chart 3 | Box plot | Row 2, right | Distribution of `ARRIVAL_DELAY` by airline | `ARRIVAL_DELAY` by `AIRLINE` |
| Table | Dataframe | Row 3 | Top 20 most delayed flights | `sort_values ARRIVAL_DELAY desc` |
| Download | Button | Footer | Export filtered data as CSV | `st.download_button` |

---

### 3.4 Screen 3 — Geographic Maps

| Element | Type | Position | Content | Data Source |
|---|---|---|---|---|
| Map 1 | Scatter geo | Main | Airports colored by delay rate | `airport_profiles` (`LATITUDE`, `LONGITUDE`) |
| Map 2 | Line geo | Tab 2 | Routes with the highest delay volume | lat/lon of origin and destination via `airports.csv` |
| Toggle | Radio | Above map| Delay rate / Average delay / Flight volume | Display control |
| Tooltip | Hover | On point | `AIRPORT`, `CITY`, `STATE`, delay rate, average delay | `airport_profiles` |
| Legend | Colorbar | Side | Green → red scale | Plotly automatic |

---

### 3.5 Screen 4 — Clusters & Anomalies

| Element | Type | Position | Content | Data Source |
|---|---|---|---|---|
| PCA Chart | 2D Scatter | Row 1 | Airports in PCA space colored by cluster | `pca_coords` + `CLUSTER` from `airport_profiles` |
| Clusters table | Dataframe | Row 2 | Average profile of each cluster | `groupby CLUSTER` in `airport_profiles` |
| Anomalies chart | Scatter | Row 3 | `DEPARTURE_DELAY` × `ARRIVAL_DELAY`, anomalies in red | `IS_ANOMALY` column from Isolation Forest |
| Anomalies table | Dataframe | Row 4 | Top 20 most anomalous flights with `AIRLINE_NAME`, origin/destination city | join with `airlines.csv` and `airports.csv` |
| Interpretation | Text | Between sections | Profile of each cluster in natural language | Generated after analysis |

---

### 3.6 Screen 5 — Flight Advisor (RAG Query)

| Element | Type | Position | Content | Behavior |
|---|---|---|---|---|
| Origin input | Text input | Form, col 1 | `ORIGIN_AIRPORT` (e.g., JFK) | Uppercase; validates against `airports.csv` |
| Destination input| Text input | Form, col 2 | `DESTINATION_AIRPORT` (e.g., LAX) | Uppercase; validates against `airports.csv` |
| Airline select | Selectbox | Form, col 3 | `AIRLINE_NAME` from `airlines.csv` | Dropdown with full names |
| Month slider | Slider | Form, col 1 | `MONTH` (1–12) | Displays month name |
| Day select | Selectbox | Form, col 2 | `DAY_OF_WEEK` | Mon–Sun |
| Time input | Number input | Form, col 3 | `SCHEDULED_DEPARTURE` (HHMM) | Validation 0–2359 |
| Text area | Text area | Form | Free question | Placeholder: "E.g., Is it worth it or is there a better time?" |
| Button | Button | Form | "🔍 Query Flight Advisor" | `on_click` → POST `/advise` |
| Risk badge | Metric | Result | LOW / MEDIUM / HIGH + % | Green / Yellow / Red |
| SHAP chart | Bar chart | Result | Top risk factors with % contribution | `shap_values` from the model |
| LLM Response | Text box | Result | Recommendation from Qwen 3 in natural language | `advice` field from the API response |
| Spinner | Loading | During call | "Analyzing route..." | `st.spinner` |

---

## 4. RAG Document Structure

### 4.1 Indexed Document Types

| Type | Est. Qty. | Granularity | Source |
|---|---|---|---|
| Route profile by airline | ~5,000 | `(ORIGIN_AIRPORT, DESTINATION_AIRPORT, AIRLINE)` | `flights.csv` + `airlines.csv` |
| Airport profile | ~350 | `ORIGIN_AIRPORT` | `airport_profiles` + `airports.csv` |
| Seasonal pattern by route| ~2,000 | `(ORIGIN_AIRPORT, DESTINATION_AIRPORT, MONTH)` | `flights.csv` |
| Pattern by day of the week | ~2,000 | `(ORIGIN_AIRPORT, DESTINATION_AIRPORT, DAY_OF_WEEK)` | `flights.csv` |
| Summary by airline | ~15 | `AIRLINE` | `flights.csv` + `airlines.csv` |

---

### 4.2 Template — Route Document

Text generated for each `(ORIGIN_AIRPORT, DESTINATION_AIRPORT, AIRLINE)`:

| Field in Text | Source Column | Generated Example |
|---|---|---|
| Header | `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`, `AIRLINE_NAME` | `"Route: JFK (New York) → LAX (Los Angeles) \| Delta Air Lines"` |
| Total flights | `count(*)` | `"Total flights analyzed: 4,832"` |
| Delay rate | `IS_DELAYED.mean()` | `"71% of flights are delayed more than 15 minutes"` |
| Average delay | `ARRIVAL_DELAY.mean()` | `"Average arrival delay: 34 minutes"` |
| Main cause | `max(AIRLINE_DELAY, WEATHER_DELAY, AIR_SYSTEM_DELAY, LATE_AIRCRAFT_DELAY)` | `"Main cause: airline operational problems"` |
| Airline delay | `AIRLINE_DELAY.mean()` | `"Operational problems: 18 min on average"` |
| Weather delay | `WEATHER_DELAY.mean()` | `"Weather: 4 min on average"` |
| Traffic delay | `AIR_SYSTEM_DELAY.mean()` | `"Air traffic control: 9 min on average"` |
| Best day | `IS_DELAYED` by `DAY_OF_WEEK` (min) | `"Best day to fly: Tuesday"` |
| Worst day | `IS_DELAYED` by `DAY_OF_WEEK` (max) | `"Worst day to fly: Friday"` |
| Best month | `IS_DELAYED` by `MONTH` (min) | `"Best month: September"` |
| Worst month | `IS_DELAYED` by `MONTH` (max) | `"Worst month: December"` |

**Document Metadata:**

| Metadata | Type | Source Column | Usage |
|---|---|---|---|
| `origin` | str | `ORIGIN_AIRPORT` | Filter by origin airport |
| `dest` | str | `DESTINATION_AIRPORT` | Filter by destination airport |
| `carrier` | str | `AIRLINE` | Filter by airline (IATA code) |
| `carrier_name` | str | `AIRLINE_NAME` via `airlines.csv` | Friendly display |
| `delay_rate` | float | `IS_DELAYED.mean()` | Filter by risk level |
| `doc_type` | str | — | `"route"` — differentiates doc types |

---

### 4.3 Template — Airport Document

Text generated for each `ORIGIN_AIRPORT`, enriched with `airports.csv`:

| Field in Text | Source Column | Generated Example |
|---|---|---|
| Header | `IATA_CODE`, `AIRPORT`, `CITY`, `STATE` | `"Airport: JFK — John F. Kennedy Intl (New York, NY)"` |
| Total flights | `count(*)` | `"Volume: 287,432 flights in the analyzed period"` |
| Delay rate | `IS_DELAYED.mean()` | `"Overall delay rate: 28% of flights"` |
| Average delay | `ARRIVAL_DELAY.mean()` | `"Average arrival delay: 42 minutes"` |
| Main bottleneck | `max(AIR_SYSTEM_DELAY, WEATHER_DELAY, AIRLINE_DELAY, LATE_AIRCRAFT_DELAY)` | `"Main bottleneck: air traffic congestion"` |
| Traffic impact | `AIR_SYSTEM_DELAY.mean()` | `"Delay by air control: 22 min on average"` |
| Weather impact | `WEATHER_DELAY.mean()` | `"Climatic impact: 11 min on average"` |
| Cluster | `CLUSTER` from `airport_profiles` | `"Profile: high-volume hub with operational bottleneck"` |

---

### 4.4 Retrieval Strategy

| Step | Description | Parameters |
|---|---|---|
| **Search 1 — Exact route** | Filter by `origin` + `dest` in FAISS metadata | `k=2`, metadata filter |
| **Search 2 — Broad context**| Free semantic search with the user's question | `k=4`, no filter |
| **Deduplication** | Remove repeated docs between the two searches | By hash of `page_content` |
| **Final context** | Concatenation of the top-4 unique docs with a `---` separator | Max. ~2,000 tokens |
| **Fallback** | If route is not found, returns docs from the origin airport | Filter by `origin` only |

---

### 4.5 Base Prompt for Qwen 3

| Prompt Section | Content | Columns Used | Est. Tokens |
|---|---|---|---|
| **System** | Persona: expert consultant on US flights | — | ~80 |
| **Flight data** | Route with city name, airline, `SCHEDULED_DEPARTURE`, `DAY_OF_WEEK`, `MONTH` | `airports.csv` `CITY`, `airlines.csv` `AIRLINE` | ~80 |
| **ML Prediction**| Probability of delay, risk level, top SHAP factors with readable names | `IS_DELAYED` prob, SHAP values | ~100 |
| **RAG Context** | Documents retrieved from the vector store (route and airport profiles) | `airport_profiles`, route docs | ~800 |
| **Instructions** | Direct tone, max 4 paragraphs, end with "My recommendation:" | — | ~80 |
| **Total input** | — | — | ~1,140 |
| **Expected output**| Natural language response with actionable recommendation | — | ~250 |

---

*Tech Challenge Phase 03 | FIAP MLET 2026*

*Author: Guilherme Lossio*
