"""
src/learning/critic.py — Hermes-Critic: Learning Loop (Phase 5)

Analyses a completed MiroFish simulation against the original Hermes message
and produces an SOP Update — a Markdown snippet that HermesEngine prepends to
its system prompt on the next run.

Persists two artefacts:
  sops/latest_feedback.md        — running append-log of every SOP produced
  sops/{scenario}_v{n}.md        — versioned file auto-loaded by HermesEngine

Usage:
    from dataclasses import asdict
    critic = CriticEngine()
    sop = critic.analyze(
        hermes_message=hermes_result.message.human_readable,
        sim_result=asdict(sim_result),
    )
    print(sop)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from .. import config
    from ..hermes.engine import _build_clients
except ImportError:
    import config  # type: ignore[no-redef]
    from hermes.engine import _build_clients  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

_SOPS_DIR = Path(__file__).resolve().parent.parent.parent / "sops"
_LATEST_FEEDBACK = _SOPS_DIR / "latest_feedback.md"

_CRITIC_SYSTEM_PROMPT = """\
You are Hermes-Critic, an after-action analyst for emergency evacuation systems.
You receive a Hermes evacuation order and the results of a MiroFish population simulation.

Your task: diagnose the communication failure and produce a concise SOP Update.

Focus on: message clarity for multi-hop word-of-mouth, route specificity,
and whether Skeptical agents had enough verifiable, self-contained information
to act without needing external confirmation.

OUTPUT FORMAT: Return a Markdown snippet only. No preamble, no trailing commentary.
Start with: ## SOP Update — [brief 3–5 word diagnosis title]
Then 1–3 bullet points, each on its own line: "- **Rule:** [one-sentence instruction for Hermes]"
Be prescriptive, not descriptive. Each rule must be actionable in the next message.

EXAMPLE OUTPUT:
## SOP Update — Skeptical Agents Need Verifiable Anchors
- **Rule:** Include at least one verifiable data point (satellite timestamp, confirmed road closure, authority name) so Skeptical agents can self-validate without a second source.
- **Rule:** State the shelter destination as a precise street intersection, not a building name alone.\
"""

_CRITIC_USER_TEMPLATE = """\
## Original Hermes Evacuation Message
{hermes_message}

## MiroFish Simulation Results
- Total agents: {total_agents}
- Evacuated (safe + en route at end): {evacuated} ({evacuation_rate:.1%})
- Informed but never acted (hesitation / skepticism): {informed_never_acted}
- Never reached by message: {never_informed}
- Stranded — Immobile agents (expected baseline): {stranded}
- Token preservation at final hop: {final_preservation:.1%}
- Bottleneck roads: {bottleneck_edges}

## Diagnosis Request
Skeptical agents (30% of population) require confirmation from 2 distinct neighbor
sources AND a token preservation rate >60% to transition to EVACUATING.
Given the metrics above, identify the primary failure mode:

1. **Framing** — the threat lacked enough authority/credibility for Skeptical self-verification
2. **Clarity** — route or shelter details degraded too fast through word-of-mouth hops
3. **Content** — missing verifiable data points that Skeptical agents need to act alone

Produce an SOP Update that Hermes must prepend to its next evacuation message.\
"""


class CriticEngine:
    """
    After-action critic for the Hermes / MiroFish pipeline.

    Sends a diagnosis prompt to the configured LLM provider and returns an SOP
    Update Markdown snippet. Automatically persists the result to two locations
    so HermesEngine picks it up on the next invocation.

    Args:
        provider:     "groq" or "anthropic". Defaults to config.LLM_PROVIDER.
        sop_scenario: Prefix used for versioned SOP files (default "valencia").
    """

    def __init__(
        self,
        provider: str | None = None,
        *,
        sop_scenario: str = "valencia",
    ) -> None:
        self.provider = provider or config.LLM_PROVIDER
        self._client, _ = _build_clients(self.provider)
        self._scenario = sop_scenario
        _SOPS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(
            "CriticEngine initialised — provider=%s  model=%s  scenario=%s",
            self.provider,
            self._client.model,
            sop_scenario,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, hermes_message: str, sim_result: dict) -> str:
        """
        Diagnose simulation failures and return an SOP Update Markdown snippet.

        The snippet is also:
        - Written to sops/{scenario}_v{n}.md (picked up by HermesEngine next run)
        - Appended to sops/latest_feedback.md (running history)

        Args:
            hermes_message: The human_readable field from the Hermes evacuation message.
            sim_result:     Dict matching SimulationResult fields. Easiest source:
                            ``dataclasses.asdict(simulation.run())``

        Returns:
            SOP Update as a Markdown string.

        Raises:
            RuntimeError: LLM provider is not configured (missing API key).
        """
        user_prompt = self._build_user_prompt(hermes_message, sim_result)
        evac_pct = sim_result.get("evacuation_rate", 0.0) * 100
        logger.info(
            "CriticEngine.analyze — run_id=%s  evacuation_rate=%.1f%%  provider=%s",
            sim_result.get("run_id", "unknown"),
            evac_pct,
            self.provider,
        )

        raw = self._client.complete(_CRITIC_SYSTEM_PROMPT, user_prompt, max_tokens=512)
        sop_update = raw.strip()

        self._persist(sop_update, sim_result)
        return sop_update

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_user_prompt(self, hermes_message: str, sim_result: dict) -> str:
        total = sim_result.get("total_agents", 0)
        evacuated = sim_result.get("evacuated", 0)
        evacuation_rate = sim_result.get("evacuation_rate", 0.0)
        informed_never_acted = sim_result.get("informed_never_acted", 0)
        never_informed = sim_result.get("never_informed", 0)
        # Stranded = Immobile agents; not tracked separately in SimulationResult
        stranded = max(0, total - evacuated - informed_never_acted - never_informed)
        decay_curve = sim_result.get("decay_curve", [1.0])
        final_preservation = decay_curve[-1] if decay_curve else 1.0
        bottleneck_edges = sim_result.get("bottleneck_edges", [])

        return _CRITIC_USER_TEMPLATE.format(
            hermes_message=hermes_message,
            total_agents=total,
            evacuated=evacuated,
            evacuation_rate=evacuation_rate,
            informed_never_acted=informed_never_acted,
            never_informed=never_informed,
            stranded=stranded,
            final_preservation=final_preservation,
            bottleneck_edges=", ".join(bottleneck_edges) if bottleneck_edges else "none recorded",
        )

    def _next_version(self) -> int:
        """Return the next integer version for this scenario's SOP files."""
        existing = sorted(_SOPS_DIR.glob(f"{self._scenario}_v*.md"))
        if not existing:
            return 1
        match = re.search(r"_v(\d+)$", existing[-1].stem)
        return int(match.group(1)) + 1 if match else 1

    def _persist(self, sop_update: str, sim_result: dict) -> None:
        """Write versioned SOP file and append an entry to latest_feedback.md."""
        version = self._next_version()
        versioned_path = _SOPS_DIR / f"{self._scenario}_v{version}.md"
        versioned_path.write_text(sop_update, encoding="utf-8")
        logger.info("SOP v%d written → %s", version, versioned_path.name)

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        run_id = sim_result.get("run_id", "unknown")
        evac_rate = sim_result.get("evacuation_rate", 0.0)

        entry = (
            f"\n---\n"
            f"<!-- run: {run_id} | ts: {timestamp} | "
            f"evac_rate: {evac_rate:.1%} | sop: {versioned_path.name} -->\n\n"
            f"{sop_update}\n"
        )
        with _LATEST_FEEDBACK.open("a", encoding="utf-8") as fh:
            fh.write(entry)
        logger.info("Appended to %s", _LATEST_FEEDBACK.name)
