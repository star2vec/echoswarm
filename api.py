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
import os
import sys
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent / "src"))

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


# ── Models ─────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    scenario: str = "paiporta"


class SatelliteRefreshRequest(BaseModel):
    date: str = "2024-10-30"
    flood_event_id: str = "live_refresh"
    threshold_db: float = -18.0


# ── Core orchestration ─────────────────────────────────────────────────────────

def _load_scenario(name: str) -> dict:
    path = _SCENARIOS_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(f"Scenario '{name}' not found (looked for {path})")
    return json.loads(path.read_text(encoding="utf-8"))


def run_orchestration(
    scenario_name: str,
    tick_callback: Callable[[dict | None], None] | None = None,
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
    n_agents        = scenario["n_agents"]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        # ── 1. Flood injection ─────────────────────────────────────────────────
        polygons  = get_flooded_sectors(source="local", path=flood_data_path)
        raw_union = unary_union(polygons)
        if raw_union.geom_type not in ("Polygon", "MultiPolygon"):
            flood_geom = MultiPolygon(
                [g for g in raw_union.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
            )
        else:
            flood_geom = raw_union

        reset_flood(flood_event_id, driver)
        inject_flood(flood_geom, flood_event_id, driver)

        # ── 2. Hermes ──────────────────────────────────────────────────────────
        ctx          = get_graph_context(sector, driver)
        hermes       = HermesEngine(sop_scenario=scenario_name)
        hermes_result = hermes.generate(ctx, sector=sector)

        # ── 3. Build swarm ─────────────────────────────────────────────────────
        G_passable, G_full = build_nx_graph(driver)
        shelter_node       = find_shelter_node(G_passable, driver)
        key_tokens         = extract_key_tokens(hermes_result)
        agents             = spawn_agents(G_full, n_agents)

        # ── 4. Simulation ──────────────────────────────────────────────────────
        config = SimulationConfig(n_agents=n_agents, max_ticks=50)
        sim    = Simulation(
            G_passable, G_full, agents, key_tokens, shelter_node, config,
            tick_callback=tick_callback,
        )
        sim_result = sim.run()

        # ── 5. Critic ──────────────────────────────────────────────────────────
        critic     = CriticEngine(sop_scenario=scenario_name)
        sop_update = critic.analyze(
            hermes_message=hermes_result.message.human_readable,
            sim_result=asdict(sim_result),
        )

        # ── 6. Geometry lookups ────────────────────────────────────────────────
        unique_node_ids = list({a.node_id for a in agents} | {shelter_node})
        node_coords     = get_node_coords(unique_node_ids, driver)

        flooded_road_ids = [r["id"] for r in ctx.get("flooded_roads", []) if r.get("id")]
        road_geom        = get_road_geometry(sim_result.bottleneck_edges, flooded_road_ids, driver)

        # ── 7. Assemble payload ────────────────────────────────────────────────
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
async def ws_run(websocket: WebSocket, scenario: str = "paiporta") -> None:
    """
    Stream simulation progress tick-by-tick, then send the final payload.

    Message types sent to client:
        {"type": "tick",     "data": {tick, safe, evacuating, informed, waiting, ...}}
        {"type": "complete", "data": <full SimulationPayload>}
        {"type": "error",    "message": "<error string>"}
    """
    await websocket.accept()
    loop  = asyncio.get_running_loop()
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def tick_cb(data: dict | None) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(data), loop)

    future = asyncio.ensure_future(
        loop.run_in_executor(None, run_orchestration, scenario, tick_cb)
    )

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            await websocket.send_json({"type": "tick", "data": item})

        payload = await future
        await websocket.send_json({"type": "complete", "data": payload})

    except Exception as exc:
        if not future.done():
            future.cancel()
        await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        await websocket.close()


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
            try:
                polygons = get_flooded_sectors_live(
                    bbox=_cfg.VALENCIA_BBOX,
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
