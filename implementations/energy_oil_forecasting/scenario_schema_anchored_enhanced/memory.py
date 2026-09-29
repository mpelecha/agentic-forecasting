"""Persistent, cross-origin scenario-framework memory.

Every call to :class:`~energy_oil_forecasting.scenario_schema_anchored_enhanced.predictor.ScenarioSchemaAnchoredEnhancedPredictor.predict`
appends the factors/scenarios the LLM produced for that origin to a JSONL
file, and reads back a few prior entries before the next call — so scenario
names and thematic structure ("Escalation continues", "OPEC+ discipline
holds", ...) persist across origins instead of the LLM re-inventing a fresh
naming scheme from a blank slate every single time it runs.

Storage is a JSONL file per (predictor_id, task_id) under
``data/scenario_memory/`` (see :data:`~energy_oil_forecasting.paths.DATA_DIR`
— gitignored, generated output, same convention as ``data/predictions/``).
Writes are append-only: no read-modify-write, so an interrupted or re-run
backtest cannot corrupt earlier entries, the same resumability discipline
``cached_multi_backtest`` uses for its own prediction cache. A predictor
re-run for the same origin simply appends a second record for that
``as_of`` — see :func:`read_prior_frameworks` for how that's resolved.

**Leakage safety.** :func:`read_prior_frameworks` only ever returns records
whose ``as_of`` is strictly before the requesting origin's ``as_of``,
regardless of write order, of which spec window (backtest vs eval) a record
came from, or of whether a later-dated record already sits in the file (from
a subsequent origin in the same run, or from a completely different, later
backtest run reusing this store). Memory is deliberately continuous across
the backtest/eval boundary — the agent should remember its own scenario
history by calendar time, not restart amnesiac at a spec-window seam — but
the strict ``<`` comparison is the same temporal fence
``scenario_schema_anchored``'s prompt already enforces for ``search_web``
(``cutoff_date`` must equal ``as_of``): a record cannot be "prior" to an
origin sharing its own date.
"""

from __future__ import annotations

import json
from pathlib import Path

from energy_oil_forecasting.paths import DATA_DIR

MEMORY_DIR = DATA_DIR / "scenario_memory"


def _memory_path(predictor_id: str, task_id: str) -> Path:
    """Return the JSONL path for this predictor/task, creating its directory."""
    safe_predictor_id = predictor_id.replace("/", "_")
    directory = MEMORY_DIR / safe_predictor_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{task_id}.jsonl"


def append_framework(
    predictor_id: str,
    task_id: str,
    *,
    as_of: str,
    factors: list[dict],
    scenarios: list[dict],
) -> None:
    """Append one origin's factors/scenarios to the memory store.

    Parameters
    ----------
    predictor_id : str
        The predictor's own ``predictor_id`` — scopes memory to one agent
        variant/model/modality combination, the same granularity
        ``data/predictions/`` already caches at.
    task_id : str
        The forecasting task's id.
    as_of : str
        The forecast origin date, ``YYYY-MM-DD``. Compared lexicographically
        by :func:`read_prior_frameworks`, so this format is load-bearing.
    factors : list of dict
        ``WtiFactor.model_dump()`` entries from this origin's LLM output.
    scenarios : list of dict
        ``WtiScenarioCard.model_dump()`` entries from this origin's LLM
        output.
    """
    record = {"as_of": as_of, "factors": factors, "scenarios": scenarios}
    path = _memory_path(predictor_id, task_id)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_prior_frameworks(
    predictor_id: str,
    task_id: str,
    *,
    before: str,
    limit: int = 2,
) -> list[dict]:
    """Return up to ``limit`` most recent frameworks strictly before ``before``.

    Parameters
    ----------
    predictor_id, task_id : str
        Same meaning as in :func:`append_framework`.
    before : str
        The requesting origin's ``as_of`` (``YYYY-MM-DD``). Only records with
        a strictly earlier ``as_of`` are returned — see the module docstring
        for why this is strict rather than ``<=``.
    limit : int, default=2
        Maximum number of frameworks to return, most recent first in time
        but returned oldest-to-newest (so the prompt reads as a timeline).
        Kept small deliberately: this is a consistency aid for the LLM's own
        prompt, not a full archive dump.

    Returns
    -------
    list of dict
        Each entry is ``{"as_of": ..., "factors": [...], "scenarios": [...]}``,
        oldest first. Empty if no file exists yet or nothing qualifies.
    """
    path = _memory_path(predictor_id, task_id)
    if not path.exists():
        return []

    candidates: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["as_of"] < before:
                candidates.append(record)

    # Last write wins per as_of: a re-run of the same origin appends a
    # second record rather than overwriting the first (append-only, see
    # module docstring), so dedupe here rather than on the write path.
    by_as_of = {record["as_of"]: record for record in candidates}
    ordered = sorted(by_as_of.values(), key=lambda record: record["as_of"])
    return ordered[-limit:]


__all__ = ["MEMORY_DIR", "append_framework", "read_prior_frameworks"]
