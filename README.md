# ECHO-SWARM

Closed-loop crisis communication optimizer. An LLM generates a structured evacuation order, a heterogeneous agent-based model simulates how that message propagates and degrades through a real road-network graph, and a second LLM pass diagnoses the failure mode and rewrites the communication policy for the next run.

Built to be **globally deployable**, but battle-tested on the Valencia DANA flood of October 29, 2024 (200+ fatalities). ECHO-SWARM is **grid-agnostic**: draw a bounding box anywhere on the globe, and the system dynamically fetches local OpenStreetMap data and live Copernicus Sentinel-1 SAR imagery to build a custom routing graph and simulate the flood in seconds.

---

## Demo

*(Add a GIF here of the agents moving on the map)*

*(Add a screenshot here of the Neo4j graph showing the flooded nodes)*

---

## What makes this different from a standard evacuation simulation

Most agent-based evacuation models treat the communication layer as a given: message is sent, agents comply or don't, done. ECHO-SWARM treats the message itself as a variable to be optimized, and uses simulation outcomes as the feedback signal.

The loop:

```
Flood state (Sentinel-1 SAR)
        │
        ▼
Neo4j road network graph  ──────────────────────────────────┐
        │                                                    │
        ▼                                                    │
Hermes (Claude + CERC protocol)                             │
  ├─ generates structured 5-W evacuation order              │
  └─ Clarity Validator (2nd LLM call) scores each W 1–10   │
        │ passes threshold (≥7/10) or retries               │
        ▼                                                    │
MiroFish agent swarm (500 agents, 50 ticks)                 │
  ├─ tick: relay → decide → move → contagion                │
  ├─ tracks token preservation per hop                      │
  └─ records bottleneck edges, unreachable nodes            │
        │                                                    │
        ▼                                                    │
Critic (Claude)                                             │
  ├─ diagnoses: framing failure / clarity degradation /     │
  │             content gap                                 │
  └─ writes SOP delta → sops/{scenario}.md ─────────────────┘
                                              (prepended to
                                             Hermes prompt
                                             next run)
```

The SOP file is append-only history. Across test runs using real Valencia data, the Critic's autonomous SOP updates improved the agent survival rate from **16% (baseline) to 81%** by identifying and patching communication bottlenecks for 'Skeptical' demographics.

---

## Agent model

Four behavioral types, fixed population shares:

| Type | % | Decision rule | Relay behavior |
|------|---|---------------|----------------|
| Compliant | 40 | Acts when token preservation > 60% | Verbatim, ~80% relay prob |
| Skeptical | 30 | Requires confirmation from **2 distinct neighbors** before acting | Drops 1 token on relay |
| Panic | 20 | Acts immediately, ignores flooded edges | Drops 1–2 tokens; infects neighbors |
| Immobile | 10 | Never acts, never relays | — |

**Dual-graph architecture:** Compliant and Skeptical agents route on `G_passable` (flood edges removed). Panic agents route on `G_full` — models the well-documented phenomenon of panic overriding hazard awareness. The graphs are computed once from Neo4j at simulation start and held in memory as NetworkX DiGraphs.

**Panic contagion:** After each movement step, Panic agents within `panic_radius=2` hops convert Compliant/Skeptical with `p=0.3`. Crucially, contagion happens *after* movement, so new panickers take effect next tick — no same-tick cascade artifacts.

**Snapshot semantics on relay:** All relay operations are collected into a buffer before any are applied. Prevents information from propagating more than one hop per tick regardless of graph topology.

**Behavioral rationale:** The Skeptical agent requiring 2 sources is a direct implementation of the two-step flow model (Katz & Lazarsfeld, 1955). It generates a realistic failure mode: if the message arrives via only one neighbor, Skeptical agents stay WAITING even if clarity is high. This is the failure mode the Critic is most likely to diagnose.

---

## Information decay

Hermes extracts key tokens from its own output: `{route_name, shelter_name, action_verb, closed_roads, ...}`. Each agent carries `tokens: frozenset[str]` — only the tokens it actually received.

```
preservation_rate(agent) = len(agent.tokens) / len(canonical_tokens)
```

Aggregated per tick across all informed agents, this produces a decay curve:

```
Tick 1  (hop 0–1):  0.98  ██████████████████████████████
Tick 5  (hop 3–5):  0.73  █████████████████████
Tick 10 (hop 7–9):  0.52  ███████████████
Tick 15 (hop 11+):  0.34  ██████████
```

A Skeptical agent at hop 15 receives a 34% message — below the 60% threshold to trigger action. They remain WAITING. The Critic reads this and tells Hermes: *the route instruction has too many tokens to survive 10+ hops. Compress it.*

This is a deliberately crude but interpretable proxy for semantic degradation. The alternative — embedding distance at each hop — would be more accurate but requires a model call per relay, which doesn't scale to 500 agents × 50 ticks.

---

## Hermes engine

Hermes is not a free-form LLM call. It is constrained to produce structured JSON satisfying the CERC (Crisis and Emergency Risk Communication) five-W protocol:

```python
{
  "who":                "...",   # target population, specific
  "what":               "...",   # action required, active voice
  "where":              "...",   # shelter with street address
  "when":               "...",   # temporal urgency
  "which_route":        "...",   # named route + named backup
  "source_justification": "...", # satellite timestamp + confidence
  "human_readable":     "..."    # full plain-language message
}
```

After generation, a **second LLM call** (Clarity Validator) scores each field 1–10. If `overall < 7` or any critical field fails, the failure note is appended to the prompt and Hermes retries. Max 2 retries before fallback.

The system prompt is **prompt-cached** (Anthropic cache breakpoint after the static CERC protocol block). The dynamic context — flooded roads, open routes, shelter coords, satellite source — is injected as JSON in the user turn. This matters: at 500-agent scale with repeated runs, uncached Hermes calls are the dominant cost.

The current SOP file for the active scenario is prepended to the system prompt before the cache breakpoint, so SOP updates don't break the cache.

---

## Critic engine

After simulation, the Critic receives:

- The original Hermes message
- Aggregated failure breakdown: `informed_never_acted`, `never_informed`, `preservation_rate` at final tick
- Bottleneck roads by crossing count
- Agent type breakdown by final state

It classifies the failure mode (framing / clarity / content) and outputs a concrete Markdown SOP delta:

```markdown
## SOP Update — Route Instruction Compression
- Name exactly one primary route (street name only, ≤4 words).
- Append one backup route on a separate line.
- Do not name closed roads in the route instruction — put closures in a separate sentence.
```

This is written to `sops/{scenario}.md` (overwrite — latest policy only) and appended to `sops/{scenario}_history.md` (full learning history). On next run, Hermes prepends the current SOP before generating.

---

## Graph layer (Neo4j)

**Dynamic Topologies:** The graph isn't hardcoded. When you select a new bounding box in the UI, the backend reaches out to the Overpass API, downloads the local road network, and builds the Neo4j routing topology from scratch in ~1.5 seconds before injecting the satellite flood polygons.

The road network is loaded from OpenStreetMap via Overpass API and stored as a directed property graph:

```
(:Intersection {id, lat, lon, sector})
  -[:CONNECTS {road_id, length_m, travel_time_min, passable: bool}]->
(:Intersection)

(:Road {id, name, highway, lanes, versioning_strategy})
(:RoadState {id, passable, cause, timestamp, flood_depth})  // audit log
```

**Flood injection is atomic:** Each road state change writes a `(:RoadState)` audit node and flips `[:CONNECTS].passable` in a single transaction. The routing query only traverses `passable=true` edges — no post-filter on shortest paths, which would silently return flood-blocked routes.

**Waterway adjacency versioning:** Roads within 50m of a waterway node use `versioning_strategy="append"` (history preserved). Interior roads use `"replace"`. Models realistic re-flooding risk on riparian roads.

**Evacuation routing:**

```cypher
MATCH path = (src)-[:CONNECTS*..40]->(shelter)
WHERE ALL(r IN relationships(path) WHERE r.passable = true)
RETURN path
ORDER BY reduce(t=0, r IN relationships(path) | t + r.travel_time_min) ASC
LIMIT 1
```

Bounded variable-length match (`*..40`) prunes during traversal — not `shortestPath()` + WHERE filter, which expands the full graph before filtering. Routes are pre-computed from every intersection to shelter at simulation start and held in memory.

---

## Satellite pipeline

Live Sentinel-1 SAR via Copernicus CDSE:

1. OAuth2 client credentials → access token
2. Sentinel Hub Process API: Automatically targets the user's selected bounding box coordinates to pull live VV-polarization backscatter (σ°) as 512×512 uint8 PNG
3. Pixel classification: `pixel < threshold_db_to_uint8(−20.0)` → flood
4. 16× downsample (512px → 32×32 cells) for noise reduction
5. Shapely `unary_union` → flood polygons
6. `inject_flood(polygons)` → Neo4j atomic write

Without CDSE credentials, falls back to pre-extracted EMS flood extent from the October 2024 DANA event (EMSR728). The fallback is not a mock — it's the actual Copernicus Emergency Management Service activation data for Paiporta.

Avoiding GDAL entirely: SAR processing runs on PIL + NumPy. `pyproj` handles CRS transforms for the bounding box. This was a deliberate dependency choice — GDAL install failures are a known failure mode on fresh machines at hackathons.

---

## Quickstart

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker.

```bash
git clone <repo>
cd echoswarm
uv sync

cp .env.example .env
# Set ANTHROPIC_API_KEY (required)
# Set CDSE_CLIENT_ID + CDSE_CLIENT_SECRET (optional, falls back to local EMS data)

docker-compose up -d          # Neo4j 5.26 on :7687, browser on :7474
# Wait ~30s for APOC to load, then:

PYTHONPATH=src uvicorn api:app --reload
```

Open `http://localhost:8000/ui/swarm`. First run loads the OSM graph (~60s, cached after). Hit **Run Mission**.

For remote access: `ngrok http 8000`. The UI auto-derives WS/API base from `window.location.origin`.

---

## API

| | Endpoint | |
|---|---|---|
| `WS` | `/ws/run?scenario=paiporta&agents=500` | Streams tick JSON + final payload |
| `GET` | `/scenarios` | List scenario names |
| `POST` | `/api/refresh_map` | Rebuild graph + inject Sentinel-1 for new bbox |
| `GET` | `/api/topology` | Road graph sample (vis.js, max 4k edges) |
| `POST` | `/run` | Async run → `run_id` |
| `GET` | `/run/{id}/result` | Completed payload |

WebSocket emits two message types:

```jsonc
// Per tick
{ "type": "tick",
  "data": { "tick": 12, "n_safe": 87, "n_evacuating": 203,
            "n_informed": 156, "n_waiting": 54, "preservation_rate": 0.61 }}

// On completion — includes hermes output, critic SOP, agent replay,
//                  bottleneck roads, flooded roads, time series
{ "type": "complete", "data": { ... }}
```

---

## Stack

| | |
|---|---|
| Agent simulation | Python, NetworkX (in-memory routing) |
| Knowledge graph | Neo4j 5.26, APOC |
| OSM ingestion | Overpass API (overpy) |
| Spatial ops | Shapely, pyproj (no GDAL) |
| Satellite | Copernicus CDSE, Sentinel Hub Process API |
| LLM | Anthropic Claude (prompt-cached Hermes + Critic) |
| API / WebSocket | FastAPI, uvicorn |
| UI | Leaflet, Chart.js, vis-network |

---

## Short description

> Agent-based flood evacuation simulator with a closed LLM feedback loop: Claude generates a CERC-structured evacuation order, a 500-agent behavioral swarm (Compliant / Skeptical / Panic / Immobile) propagates it through a live Neo4j road-network graph while tracking message token decay per hop, and a Critic LLM diagnoses the failure mode and rewrites the communication SOP for the next run. Sentinel-1 SAR flood data via Copernicus CDSE. Built to be globally deployable across dynamic coordinate bounding boxes.
