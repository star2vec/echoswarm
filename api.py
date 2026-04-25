"""
ECHO-SWARM FastAPI bridge — Phase 6 UI layer.

Usage:
    PYTHONPATH=src uvicorn api:app --reload

Endpoints:
    GET  /scenarios               — list available scenario names
    WS   /ws/run?scenario=NAME    — streaming: tick-by-tick + final payload
    POST /run  {"scenario": NAME} — async polling: returns 202 with run_id
    GET  /run/{run_id}/status     — poll progress
    GET  /run/{run_id}/result     — fetch completed payload

The orchestration function is engine-agnostic: swap Python MiroFish for C++
ECS by changing what produces SimulationResult — the JSON contract is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


class _SafeJSONEncoder(json.JSONEncoder):
    """Handles numpy scalars/arrays and other non-native JSON types that can
    appear in networkx/simulation results, so serialization errors surface as
    clear log messages rather than silent WebSocket drops."""

    def default(self, obj: object) -> object:
        # numpy types — import guarded so numpy is optional
        try:
            import numpy as np  # noqa: PLC0415
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        # Fallback for anything with a standard numeric coercion
        if hasattr(obj, "__index__"):
            return int(obj)
        if hasattr(obj, "__float__"):
            return float(obj)
        return super().default(obj)

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger as _llog
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase
from pydantic import BaseModel
from shapely import unary_union
from shapely.geometry import MultiPolygon

load_dotenv()

from graph.queries import (
    get_graph_context,
    get_node_coords,
    get_road_geometry,
    inject_flood,
    reset_flood,
)
from hermes.engine import HermesEngine
from learning.critic import CriticEngine
from satellite.local import get_flooded_sectors
from satellite.flood_engine import CDSEUnavailableError, get_flooded_sectors_live
import config as _cfg
from swarm.agents import AgentState
from swarm.simulation import (
    Simulation,
    SimulationConfig,
    build_nx_graph,
    extract_key_tokens,
    find_shelter_node,
    spawn_agents,
)
from bridge.payload import build_payload

# ── Config ─────────────────────────────────────────────────────────────────────

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "echoswarm")

_SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="ECHO-SWARM API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-process store for polling-based runs: run_id → state dict
_runs: dict[str, dict] = {}

# Session-level caches to skip redundant flood work on repeated simulation runs.
# Cleared by /satellite/refresh so a manual satellite update forces re-injection.
_flood_union_cache: dict[str, Any] = {}   # flood_data_path → merged Shapely geometry
_flood_injected: dict[str, str]    = {}   # flood_event_id  → flood_data_path last injected with

# Abort flag: set by ws_run when a new connection arrives so a stale thread
# from a previous (disconnected) WebSocket stops as soon as possible.
_abort_event = threading.Event()


# ── Models ─────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    scenario: str = "paiporta"


class SatelliteRefreshRequest(BaseModel):
    date: str = "2024-10-30"
    flood_event_id: str = "live_refresh"
    threshold_db: float = -18.0
    # [min_lon, min_lat, max_lon, max_lat] WGS-84; falls back to config.VALENCIA_BBOX
    bbox: list[float] | None = None


# ── Core orchestration ─────────────────────────────────────────────────────────

def _load_scenario(name: str) -> dict:
    path = _SCENARIOS_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Scenario '{name}' not found (looked for {path})")
    return json.loads(path.read_text(encoding="utf-8"))


def run_orchestration(
    scenario_name: str,
    tick_callback: Callable[[dict | None], None] | None = None,
    n_agents_override: int | None = None,
) -> dict:
    """
    Full pipeline: flood injection → Hermes → MiroFish → Critic → payload.

    Calls tick_callback(dict) after each simulation tick so callers can stream
    progress.  Calls tick_callback(None) as a sentinel when complete.

    Blocking — run in a thread pool from async contexts.
    """
    scenario = _load_scenario(scenario_name)

    sector          = scenario["sector"]
    flood_event_id  = scenario["flood_event_id"]
    flood_data_path = scenario["flood_data_path"]
    n_agents        = n_agents_override if n_agents_override is not None else scenario["n_agents"]

    _abort_event.clear()
    t_start = time.monotonic()

    def _elapsed() -> str:
        return f"{time.monotonic() - t_start:.1f}s"

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        # ── 1. Flood injection ─────────────────────────────────────────────────
        # Cache the unary_union — merging 1117 EMSR773 polygons takes 5–30 s.
        # Cache is keyed by path and lives for the server session; cleared by
        # /satellite/refresh so a manual update still forces re-injection.
        if flood_data_path not in _flood_union_cache:
            _llog.info("[{}] building flood union from {} …", _elapsed(), flood_data_path)
            polygons  = get_flooded_sectors(source="local", path=flood_data_path)
            raw_union = unary_union(polygons)
            if raw_union.geom_type not in ("Polygon", "MultiPolygon"):
                raw_union = MultiPolygon(
                    [g for g in raw_union.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
                )
            _flood_union_cache[flood_data_path] = raw_union
            _llog.info("[{}] flood union cached", _elapsed())
        flood_geom = _flood_union_cache[flood_data_path]

        # Skip Neo4j reset+inject when flood state is already current for this
        # scenario — the graph doesn't change between runs.
        if _flood_injected.get(flood_event_id) != flood_data_path:
            reset_flood(flood_event_id, driver)
            inject_flood(flood_geom, flood_event_id, driver)
            _flood_injected[flood_event_id] = flood_data_path
        else:
            _llog.info("[{}] flood injection skipped (cached)", _elapsed())

        if _abort_event.is_set():
            raise RuntimeError("run aborted — new connection arrived")

        # ── 2. Hermes ──────────────────────────────────────────────────────────
        _llog.info("[{}] get_graph_context …", _elapsed())
        ctx          = get_graph_context(sector, driver)
        _llog.info("[{}] hermes.generate …", _elapsed())
        hermes       = HermesEngine(sop_scenario=scenario_name)
        hermes_result = hermes.generate(ctx, sector=sector)
        _llog.info("[{}] hermes done", _elapsed())

        if _abort_event.is_set():
            raise RuntimeError("run aborted — new connection arrived")

        # ── 3. Build swarm ─────────────────────────────────────────────────────
        _llog.info("[{}] build_nx_graph …", _elapsed())
        G_passable, G_full = build_nx_graph(driver)
        _llog.info("[{}] graph: {} nodes, {} edges (passable)", _elapsed(),
                   G_passable.number_of_nodes(), G_passable.number_of_edges())
        shelter_node       = find_shelter_node(G_passable, driver)
        key_tokens         = extract_key_tokens(hermes_result)
        agents             = spawn_agents(G_full, n_agents)
        _llog.info("[{}] agents spawned ({})", _elapsed(), len(agents))

        if _abort_event.is_set():
            raise RuntimeError("run aborted — new connection arrived")

        # ── 4. Simulation ──────────────────────────────────────────────────────
        _llog.info("[{}] simulation init + dijkstra …", _elapsed())
        config = SimulationConfig(n_agents=n_agents, max_ticks=100)
        sim    = Simulation(
            G_passable, G_full, agents, key_tokens, shelter_node, config,
            tick_callback=tick_callback,
        )
        _llog.info("[{}] simulation running …", _elapsed())
        sim_result = sim.run()
        _llog.info("[{}] simulation done ({} ticks, {} safe)", _elapsed(),
                   sim_result.ticks_run, sim_result.evacuated)

        # ── 5. Critic ──────────────────────────────────────────────────────────
        _llog.info("[{}] critic …", _elapsed())
        critic     = CriticEngine(sop_scenario=scenario_name)
        sop_update = critic.analyze(
            hermes_message=hermes_result.message.human_readable,
            sim_result=asdict(sim_result),
        )
        _llog.info("[{}] critic done", _elapsed())

        # ── 6. Geometry lookups ────────────────────────────────────────────────
        _llog.info("[{}] geometry lookups …", _elapsed())
        unique_node_ids = list({a.node_id for a in agents} | {shelter_node})
        node_coords     = get_node_coords(unique_node_ids, driver)

        flooded_road_ids = [r["id"] for r in ctx.get("flooded_roads", []) if r.get("id")]
        road_geom        = get_road_geometry(sim_result.bottleneck_edges, flooded_road_ids, driver)

        # ── 7. Assemble payload ────────────────────────────────────────────────
        _llog.info("[{}] build_payload …", _elapsed())
        payload = build_payload(
            scenario_name=scenario_name,
            hermes_result=hermes_result,
            sim_result=sim_result,
            agents=agents,
            node_coords=node_coords,
            road_geom=road_geom,
            graph_context=ctx,
            sop_update=sop_update,
            shelter_node=shelter_node,
        )

    finally:
        driver.close()

    _llog.info("[{}] orchestration complete — sending payload", _elapsed())
    # Sentinel: signals WebSocket/polling that orchestration is done
    if tick_callback is not None:
        tick_callback(None)

    return payload


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/scenarios")
def list_scenarios() -> list[str]:
    """Return the names of all available scenario JSON files."""
    return sorted(p.stem for p in _SCENARIOS_DIR.glob("*.json"))


@app.websocket("/ws/run")
async def ws_run(
    websocket: WebSocket,
    scenario: str = "paiporta",
    agents: int | None = None,
) -> None:
    """
    Stream simulation progress tick-by-tick, then send the final payload.

    Query params:
        scenario: scenario name (default "paiporta")
        agents:   override n_agents from scenario JSON (1–10000)

    Message types sent to client:
        {"type": "tick",     "data": {tick, safe, evacuating, informed, waiting, ...}}
        {"type": "complete", "data": <full SimulationPayload>}
        {"type": "error",    "message": "<error string>"}
    """
    # Signal any in-progress run from a previous (now-disconnected) WebSocket
    # to stop at its next abort checkpoint.  Prevents two threads fighting over
    # the same Groq API quota when the browser reconnects mid-simulation.
    _abort_event.set()

    await websocket.accept()
    loop  = asyncio.get_running_loop()
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    # Clamp agent count to a safe range
    n_agents_override = max(1, min(10_000, agents)) if agents is not None else None

    def tick_cb(data: dict | None) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(data), loop)

    future = asyncio.ensure_future(
        loop.run_in_executor(None, run_orchestration, scenario, tick_cb, n_agents_override)
    )

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            await websocket.send_json({"type": "tick", "data": item})

        payload = await future

        # Cap agents_final to 1000 sampled entries to prevent browser UI freezing
        # on large simulations while keeping enough density for meaningful map rendering.
        agents_final = payload.get("map", {}).get("agents_final", [])
        if len(agents_final) > 1000:
            payload["map"]["agents_final"] = random.sample(agents_final, 1000)

        # Pre-serialize with a tolerant encoder so any type error surfaces as a
        # clear terminal log rather than a silent WebSocket drop.
        try:
            raw = json.dumps({"type": "complete", "data": payload}, cls=_SafeJSONEncoder)
        except Exception as serial_exc:
            _log.error("Payload serialization failed: %r", serial_exc)
            raise

        await websocket.send_text(raw)
        # Give the OS network buffer time to flush the full payload to the
        # client before the close frame is sent.  Ruled-out once confirmed
        # not the cause; harmless either way.
        await asyncio.sleep(0.5)

    except Exception as exc:
        _log.error("WebSocket run failed (%s): %r", type(exc).__name__, exc)
        if not future.done():
            future.cancel()
        # Guard the error send — if the connection is already broken this would
        # otherwise raise a second uncaught exception and bury the original one.
        try:
            await websocket.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except (RuntimeError, Exception):
            pass  # client already disconnected; nothing to close


@app.post("/run", status_code=202)
async def post_run(body: RunRequest) -> dict:
    """
    Start a simulation run asynchronously.  Returns a run_id for polling.
    Poll GET /run/{run_id}/status, then fetch GET /run/{run_id}/result.
    """
    run_id = str(uuid.uuid4())[:8]
    _runs[run_id] = {"status": "running", "ticks_done": 0, "max_ticks": 50, "payload": None}

    loop = asyncio.get_running_loop()

    def progress_cb(data: dict | None) -> None:
        if data is not None:
            _runs[run_id]["ticks_done"] = data.get("tick", 0)

    async def _task() -> None:
        try:
            payload = await loop.run_in_executor(
                None, run_orchestration, body.scenario, progress_cb
            )
            _runs[run_id].update(status="complete", payload=payload)
        except Exception as exc:
            _runs[run_id].update(status="failed", error=str(exc))

    asyncio.create_task(_task())
    return {"run_id": run_id}


@app.get("/run/{run_id}/status")
def get_run_status(run_id: str) -> dict:
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")
    state = _runs[run_id]
    return {
        "run_id":     run_id,
        "status":     state["status"],
        "ticks_done": state["ticks_done"],
        "max_ticks":  state["max_ticks"],
        **({"error": state["error"]} if state.get("error") else {}),
    }


@app.get("/run/{run_id}/result")
def get_run_result(run_id: str) -> dict:
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")
    state = _runs[run_id]
    if state["status"] != "complete":
        raise HTTPException(status_code=409, detail=f"Run status is '{state['status']}', not 'complete'")
    return state["payload"]


@app.post("/satellite/refresh")
async def satellite_refresh(body: SatelliteRefreshRequest) -> dict:
    """
    Fetch a live Sentinel-1 flood mask from CDSE and inject it into the graph.

    Falls back to the pre-loaded EMS flood data if CDSE credentials are missing
    or the Process API is unavailable — the demo always works.

    Returns:
        {"status": "live"|"fallback", "source": str, "polygons_detected": int, "edges_blocked": int}
    """
    loop = asyncio.get_running_loop()

    def _run() -> dict:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            source_label = "live"
            effective_bbox = tuple(body.bbox) if body.bbox and len(body.bbox) == 4 else _cfg.VALENCIA_BBOX
            try:
                polygons = get_flooded_sectors_live(
                    bbox=effective_bbox,
                    target_date=body.date,
                    client_id=_cfg.CDSE_CLIENT_ID,
                    client_secret=_cfg.CDSE_CLIENT_SECRET,
                    threshold_db=body.threshold_db,
                )
            except CDSEUnavailableError as exc:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "CDSE unavailable (%s) — falling back to local EMS data", exc
                )
                polygons = get_flooded_sectors(source="local")
                source_label = "fallback"

            reset_flood(body.flood_event_id, driver)

            total_edges = 0
            for polygon in polygons:
                total_edges += inject_flood(polygon, body.flood_event_id, driver)

            # Invalidate session caches so the next run_orchestration call
            # recomputes the union and re-injects the new flood state.
            _flood_union_cache.clear()
            _flood_injected.clear()

            return {
                "status":            source_label,
                "source":            "sentinel-1-cdse" if source_label == "live" else "copernicus-ems-local",
                "date":              body.date,
                "polygons_detected": len(polygons),
                "edges_blocked":     total_edges,
            }
        finally:
            driver.close()

    return await loop.run_in_executor(None, _run)
