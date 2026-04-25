"""
src/hermes/engine.py — Hermes Communication Engine (Phase 3)

Reads the live Neo4j graph state and generates CERC-compliant evacuation orders.
Every output is validated by a second LLM call (Clarity Validator) before release
to the MiroFish swarm.

Provider is configured in src/config.py:
    LLM_PROVIDER = "groq"       → Groq API, llama-3.1-70b-versatile  (default: dev/sim)
    LLM_PROVIDER = "anthropic"  → Anthropic API, claude-sonnet-4-6   (production swap)

Pluggable architecture: both providers implement the LLMClient protocol. Swapping
providers requires only the config change above — no engine code changes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

try:
    from .. import config          # installed as src.hermes.engine
except ImportError:
    import config  # type: ignore[no-redef]  # PYTHONPATH=src (hermes is top-level)

logger = logging.getLogger(__name__)

_SOPS_DIR = Path(__file__).resolve().parent.parent.parent / "sops"


# ─────────────────────────────────────────────────────────────────────────────
# Public output types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HermesMessage:
    who: str
    what: str
    where: str
    when: str
    which_route: str
    source_justification: str
    human_readable: str

    def to_dict(self) -> dict:
        return {
            "who": self.who,
            "what": self.what,
            "where": self.where,
            "when": self.when,
            "which_route": self.which_route,
            "source_justification": self.source_justification,
            "human_readable": self.human_readable,
        }


@dataclass
class ClarityScore:
    who: int
    what: int
    where: int
    when: int
    which_route: int
    overall: int
    passed: bool


@dataclass
class HermesResult:
    message: HermesMessage
    clarity: ClarityScore
    attempts: int
    provider: str
    model: str


# ─────────────────────────────────────────────────────────────────────────────
# LLM Provider protocol + implementations
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        ...


class GroqClient:
    """Groq chat completion client. Used for dev and simulation runs."""

    def __init__(self, model: str) -> None:
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("groq package not installed. Run: uv add groq") from exc
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        self._client = Groq(api_key=config.GROQ_API_KEY)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""


class AnthropicClient:
    """
    Anthropic client with optional prompt caching on the static system prompt.

    The CERC system prompt is sent with cache_control=ephemeral so repeated
    Hermes calls within the same 5-minute window hit the cache. The user prompt
    (graph context) is dynamic per call and is never cached.
    """

    def __init__(self, model: str, *, use_cache: bool = True) -> None:
        try:
            import anthropic as _anthropic
            self._anthropic = _anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic package not installed. Run: uv add anthropic") from exc
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        self._client = _anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = model
        self._use_cache = use_cache

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        if self._use_cache:
            system_param = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            system_param = system

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_param,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


def _build_clients(provider: str) -> tuple[LLMClient, LLMClient]:
    """Return (main_client, fast_client) for the configured provider."""
    if provider == "groq":
        return (
            GroqClient(config.GROQ_MAIN_MODEL),
            GroqClient(config.GROQ_FAST_MODEL),
        )
    if provider == "anthropic":
        return (
            AnthropicClient(config.ANTHROPIC_MAIN_MODEL, use_cache=True),
            AnthropicClient(config.ANTHROPIC_FAST_MODEL, use_cache=False),
        )
    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        "Set LLM_PROVIDER to 'groq' or 'anthropic' in config.py or .env."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Static prompts (CERC template — cached on Anthropic, reused on Groq)
# ─────────────────────────────────────────────────────────────────────────────

_CERC_SYSTEM_PROMPT = """\
You are Hermes, an emergency crisis communication system.
Your role is to generate evacuation orders following the CERC (Crisis and Emergency Risk Communication) protocol.

THREE PILLARS you must always follow:
1. FRAMING: Validate the threat first. Explain WHY action is needed. Never minimize.
2. CLARITY: Answer Who, What, Where, When, Which route. No bureaucratic language. No passive voice.
3. CONTENT: Always cite the data source. "Satellite imagery confirms X" increases compliance.

OUTPUT FORMAT: Return valid JSON matching this exact schema. No markdown, no prose outside the JSON.

{
  "who": "Target population — district, sector, or street name",
  "what": "Required action in direct active voice. Example: 'Evacuate immediately.'",
  "where": "Specific destination with address and capacity if known",
  "when": "Urgency level — use 'NOW', 'IMMEDIATELY', or a specific time window",
  "which_route": "Named primary route + named alternative. Include highway status if relevant.",
  "source_justification": "Data citation: satellite source, timestamp, and confirmed breach or event",
  "human_readable": "Full plain-language message combining all fields above. 2-3 sentences max."
}

BAD example: "Inhabitants should seek higher ground."
GOOD example: "Residents of Ruzafa district: Evacuate NOW. Take Calle Cuba north to the Community Center at Calle Sueca 45 (capacity 500). Highway V-30 is closed — use Avenida del Puerto instead. Sentinel-1 confirms river breach at Puente de Aragón as of 06:32 UTC."\
"""

_USER_PROMPT_TEMPLATE = """\
Current crisis state:
{graph_context_json}

Generate a CERC-compliant evacuation order for the affected population.
Return JSON only. No additional text.\
"""

_VALIDATOR_SYSTEM_PROMPT = """\
You are a Crisis Communication Auditor. Score evacuation messages on specificity and actionability.
Return JSON only. No additional text.\
"""

_VALIDATOR_USER_TEMPLATE = """\
Score the following evacuation message on the 5 W's.

Message:
{hermes_output_json}

Score each dimension 1-10:
- Who (is the target population clearly identified?)
- What (is the required action unambiguous?)
- Where (is the destination specific and navigable?)
- When (is urgency communicated?)
- Which route (is a specific route named with alternatives noted?)

Return JSON: {{"who": int, "what": int, "where": int, "when": int, "which_route": int, "overall": int, "pass": bool}}
Pass threshold: overall >= 7.\
"""

_RETRY_NOTE_TEMPLATE = (
    "Previous attempt scored {score}/10. Improve specificity on: {failed}."
)


# ─────────────────────────────────────────────────────────────────────────────
# SOP loader (Learning Loop, Phase 5)
# ─────────────────────────────────────────────────────────────────────────────

def _load_latest_sop(scenario: str) -> str:
    """
    Prepend the current SOP playbook to the system prompt.
    Returns empty string if no playbook exists yet (Phase 3 default).
    The playbook is written (overwritten) by CriticEngine to sops/{scenario}.md.
    """
    playbook = _SOPS_DIR / f"{scenario}.md"
    if not playbook.exists():
        return ""
    content = playbook.read_text(encoding="utf-8").strip()
    logger.info("Loaded SOP modifier: %s", playbook.name)
    return content


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_fences(raw: str) -> str:
    """Remove markdown code fences that some models insert despite instructions."""
    raw = raw.strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    return "\n".join(line for line in lines if not line.startswith("```")).strip()


def _extract_json_block(raw: str) -> str | None:
    """Find the first complete {...} block in raw text."""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    return raw[start:end]


def _parse_message_json(raw: str) -> HermesMessage | None:
    """Parse the 5 W's JSON from an LLM response. Tolerates fences and stray text."""
    text = _strip_fences(raw)
    data = None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        block = _extract_json_block(text)
        if block is None:
            logger.error("No JSON object in Hermes response: %.200s", raw)
            return None
        try:
            data = json.loads(block)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse failed: %s | raw: %.200s", exc, raw)
            return None

    required = {"who", "what", "where", "when", "which_route", "source_justification", "human_readable"}
    missing = required - data.keys()
    if missing:
        logger.warning("Hermes JSON missing fields: %s — filling with empty strings", missing)
        for key in missing:
            data[key] = ""

    return HermesMessage(
        who=str(data.get("who", "")),
        what=str(data.get("what", "")),
        where=str(data.get("where", "")),
        when=str(data.get("when", "")),
        which_route=str(data.get("which_route", "")),
        source_justification=str(data.get("source_justification", "")),
        human_readable=str(data.get("human_readable", "")),
    )


def _parse_clarity_json(raw: str) -> ClarityScore | None:
    """Parse the Clarity Validator JSON response."""
    text = _strip_fences(raw)
    data = None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        block = _extract_json_block(text)
        if block is None:
            logger.error("No JSON in validator response: %.200s", raw)
            return None
        try:
            data = json.loads(block)
        except json.JSONDecodeError as exc:
            logger.error("Clarity JSON parse failed: %s | raw: %.200s", exc, raw)
            return None

    dims = ["who", "what", "where", "when", "which_route"]
    scores = {d: int(data.get(d, 5)) for d in dims}
    overall = int(data.get("overall", sum(scores.values()) // len(dims)))
    # Compute pass/fail ourselves — never trust the LLM's "pass" field.
    # The LLM returns "pass": false even on 8/10 scores using its own criteria.
    passed = overall >= config.HERMES_CLARITY_PASS_THRESHOLD

    return ClarityScore(
        who=scores["who"],
        what=scores["what"],
        where=scores["where"],
        when=scores["when"],
        which_route=scores["which_route"],
        overall=overall,
        passed=passed,
    )


def _failed_dimensions(score: ClarityScore) -> list[str]:
    """Return dimension names that individually scored below the pass threshold."""
    threshold = config.HERMES_CLARITY_PASS_THRESHOLD
    return [
        name for name, val in [
            ("who", score.who),
            ("what", score.what),
            ("where", score.where),
            ("when", score.when),
            ("which_route", score.which_route),
        ]
        if val < threshold
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Hermes Engine
# ─────────────────────────────────────────────────────────────────────────────

class HermesEngine:
    """
    Reads graph context and generates CERC-compliant evacuation orders.

    Architecture: pluggable LLM provider (GroqClient / AnthropicClient).
    Main call generates the 5 W's message; fast call runs the Clarity Validator.
    Regenerates up to HERMES_MAX_RETRIES times if the validator rejects the output.

    Usage:
        engine = HermesEngine()                    # uses config.LLM_PROVIDER
        engine = HermesEngine(provider="anthropic") # override for this instance

        context = get_graph_context("Ruzafa", driver)
        result  = engine.generate(context, sector="Ruzafa")
        print(result.message.human_readable)
    """

    def __init__(
        self,
        provider: str | None = None,
        *,
        sop_scenario: str = "valencia",
    ) -> None:
        self.provider = provider or config.LLM_PROVIDER
        self._main, self._fast = _build_clients(self.provider)
        self._sop = _load_latest_sop(sop_scenario)
        logger.info(
            "HermesEngine initialised — provider=%s  main=%s  fast=%s",
            self.provider, self._main.model, self._fast.model,
        )

    def _system_prompt(self) -> str:
        if self._sop:
            return f"{self._sop}\n\n{_CERC_SYSTEM_PROMPT}"
        return _CERC_SYSTEM_PROMPT

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, graph_context: dict, *, sector: str = "unknown") -> HermesResult:
        """
        Generate and validate a CERC-compliant evacuation order from graph state.

        Args:
            graph_context: Dict returned by get_graph_context(sector, driver).
                           Contains flooded_roads, shelters, passable routes, etc.
            sector:        Human-readable sector name used only for logging.

        Returns:
            HermesResult with the validated message, clarity scores, attempt count,
            and provider/model metadata.

        Raises:
            RuntimeError: All HERMES_MAX_RETRIES attempts failed Clarity Validator.
                          The caller should log this and queue for human review.
        """
        system = self._system_prompt()
        base_user = _USER_PROMPT_TEMPLATE.format(
            graph_context_json=json.dumps(graph_context, indent=2, default=str)
        )

        # Sentinel so the retry-note branch can safely reference last_clarity
        last_clarity = ClarityScore(0, 0, 0, 0, 0, 0, False)

        for attempt in range(1, config.HERMES_MAX_RETRIES + 1):
            logger.info(
                "Hermes attempt %d/%d (sector=%s, provider=%s)",
                attempt, config.HERMES_MAX_RETRIES, sector, self.provider,
            )

            user_prompt = base_user
            if attempt > 1:
                failed = _failed_dimensions(last_clarity)
                note = _RETRY_NOTE_TEMPLATE.format(
                    score=last_clarity.overall,
                    failed=", ".join(failed) if failed else "overall specificity",
                )
                user_prompt = f"{note}\n\n{base_user}"

            raw_message = self._main.complete(system, user_prompt, config.HERMES_MAX_TOKENS)
            logger.debug("Raw Hermes output (attempt %d): %.400s", attempt, raw_message)

            message = _parse_message_json(raw_message)
            if message is None:
                logger.warning("Attempt %d: message JSON parse failed — retrying", attempt)
                continue

            clarity = self._run_validator(message)
            if clarity is None:
                logger.warning("Attempt %d: clarity JSON parse failed — retrying", attempt)
                continue

            last_clarity = clarity
            logger.info(
                "Attempt %d clarity: who=%d what=%d where=%d when=%d route=%d overall=%d → %s",
                attempt,
                clarity.who, clarity.what, clarity.where, clarity.when, clarity.which_route,
                clarity.overall, "PASS" if clarity.passed else "FAIL",
            )

            if clarity.passed:
                return HermesResult(
                    message=message,
                    clarity=clarity,
                    attempts=attempt,
                    provider=self.provider,
                    model=self._main.model,
                )

        logger.error(
            "Hermes exhausted all %d attempts for sector=%s. "
            "Best overall clarity: %d/10. Queuing for human review.",
            config.HERMES_MAX_RETRIES, sector, last_clarity.overall,
        )
        raise RuntimeError(
            f"Hermes: all {config.HERMES_MAX_RETRIES} attempts failed Clarity Validator "
            f"(best overall score: {last_clarity.overall}/10, sector={sector}). "
            "Human review required."
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_validator(self, message: HermesMessage) -> ClarityScore | None:
        """Send the generated message to the fast model for clarity scoring."""
        message_json = json.dumps(message.to_dict(), indent=2)
        raw = self._fast.complete(
            _VALIDATOR_SYSTEM_PROMPT,
            _VALIDATOR_USER_TEMPLATE.format(hermes_output_json=message_json),
            config.HERMES_FAST_MAX_TOKENS,
        )
        logger.debug("Raw clarity response: %.300s", raw)
        return _parse_clarity_json(raw)
