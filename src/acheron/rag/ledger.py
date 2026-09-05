"""Experiment ledger — persistent log of discovery loop findings, and
predicted-vs-actual tracking for MODE 6 (prediction) results.

Predicted-vs-actual design: `record_prediction()` logs a MODE 6 prediction
(organism, perturbation_gene, confidence score/tier, cited-vs-total
parameter counts, and a human-readable summary) the moment it's made, using
only plain values -- never the `PredictionModeResult`/`GRNSimulationResult`
objects themselves, so this module stays a light leaf dependency and does
not import `rag/hypothesis_engine.py` (the reverse: hypothesis_engine.py or
cli.py imports ledger.py, not the other way around). `record_actual_outcome()`
then attaches a real, manually-entered experimental result to that SAME
entry later, preserving the original `timestamp` as proof the prediction was
made before the outcome was known. `acheron report --accuracy` (see
`compute_accuracy_report()`) computes accuracy ONLY over entries that went
through both calls -- see that function's own docstring for why no other
number may be substituted for it.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from acheron.config import get_settings
from acheron.models import DiscoveryResult, LedgerEntry

logger = logging.getLogger(__name__)


class ExperimentLedger:
    """Append-only ledger for recording discovery loop outputs.

    Stores entries as individual JSON files in a dedicated directory,
    enabling both programmatic access and human review.
    """

    def __init__(self, ledger_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.ledger_dir = ledger_dir or settings.data_dir / "ledger"
        self.ledger_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        discovery: DiscoveryResult,
        notes: str = "",
        tags: Optional[list[str]] = None,
    ) -> LedgerEntry:
        """Record a discovery result as a ledger entry."""
        entry_id = f"ledger-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        entry = LedgerEntry(
            entry_id=entry_id,
            query=discovery.query,
            evidence_summary="\n".join(discovery.evidence) if discovery.evidence else "",
            hypotheses=discovery.hypotheses,
            variables=discovery.variables,
            source_ids=[s.paper_id for s in discovery.sources],
            notes=notes,
            tags=tags or [],
        )

        dest = self.ledger_dir / f"{entry_id}.json"
        dest.write_text(entry.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Ledger entry recorded: %s", entry_id)
        return entry

    def list_entries(self, tag: Optional[str] = None) -> list[LedgerEntry]:
        """List all ledger entries, optionally filtered by tag."""
        entries = []
        for path in sorted(self.ledger_dir.glob("ledger-*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entry = LedgerEntry(**data)
                if tag and tag not in entry.tags:
                    continue
                entries.append(entry)
            except Exception:
                logger.debug("Skipping malformed ledger entry: %s", path.name)
        return entries

    def get_entry(self, entry_id: str) -> Optional[LedgerEntry]:
        """Retrieve a single ledger entry by ID."""
        path = self.ledger_dir / f"{entry_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return LedgerEntry(**data)

    def search_entries(self, keyword: str) -> list[LedgerEntry]:
        """Search ledger entries by keyword in query or evidence summary."""
        keyword_lower = keyword.lower()
        results = []
        for entry in self.list_entries():
            if (
                keyword_lower in entry.query.lower()
                or keyword_lower in entry.evidence_summary.lower()
                or keyword_lower in entry.notes.lower()
            ):
                results.append(entry)
        return results

    def count(self) -> int:
        """Return the number of ledger entries."""
        return len(list(self.ledger_dir.glob("ledger-*.json")))

    def export_all(self, dest: Path) -> int:
        """Export all ledger entries to a single JSON file."""
        entries = self.list_entries()
        data = [e.model_dump(mode="json") for e in entries]
        dest.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return len(data)

    # ==================================================================
    # Predicted-vs-actual tracking (MODE 6 predictions)
    # ==================================================================
    def record_prediction(
        self,
        organism: str,
        perturbation_gene: str,
        confidence_score: float,
        confidence_tier: str,
        cited_params: int,
        total_params: int,
        prediction_summary: str,
        notes: str = "",
        tags: Optional[list[str]] = None,
    ) -> LedgerEntry:
        """Log a MODE 6 prediction BEFORE the real test runs.

        Takes plain values (not a PredictionModeResult) so this module
        never needs to import rag/hypothesis_engine.py -- the caller
        (cli.py, or hypothesis_engine.py itself) is expected to pass
        `result.overall_confidence_score`, `result.grn.n_cited_edges +
        (1 if cited bioelectric mapping else 0)`, etc., and
        `format_prediction_context(result)` as `prediction_summary`.
        """
        entry_id = f"ledger-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        entry = LedgerEntry(
            entry_id=entry_id,
            query=f"PREDICTION: {organism} / {perturbation_gene} knockdown",
            entry_type="prediction",
            organism=organism,
            perturbation_gene=perturbation_gene,
            prediction_summary=prediction_summary,
            predicted_confidence_score=confidence_score,
            predicted_confidence_tier=confidence_tier,
            predicted_cited_params=cited_params,
            predicted_total_params=total_params,
            notes=notes,
            tags=tags or [],
        )

        dest = self.ledger_dir / f"{entry_id}.json"
        dest.write_text(entry.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Prediction ledger entry recorded: %s", entry_id)
        return entry

    def record_actual_outcome(
        self,
        entry_id: str,
        actual_outcome: str,
        match: bool,
        force: bool = False,
    ) -> LedgerEntry:
        """Attach a real, manually-entered experimental outcome to a
        previously-logged prediction entry.

        Refuses entries that were never a logged prediction in the first
        place (`entry_type != "prediction"`), so accuracy reporting can
        never accidentally mix in a discovery-loop entry. Refuses to
        silently overwrite an already-recorded outcome unless
        `force=True` -- this ledger's entire value as an investor-facing
        metric depends on `actual_outcome`/`match` never being quietly
        rewritten after the fact.
        """
        entry = self.get_entry(entry_id)
        if entry is None:
            raise ValueError(f"No ledger entry found with id '{entry_id}'")
        if entry.entry_type != "prediction":
            raise ValueError(
                f"Entry '{entry_id}' is a '{entry.entry_type}' entry, not a "
                f"prediction -- only entries created by record_prediction() "
                f"can have an actual outcome attached."
            )
        if entry.actual_outcome is not None and not force:
            raise ValueError(
                f"Entry '{entry_id}' already has an actual outcome recorded "
                f"(entered {entry.actual_outcome_entered_at}) -- pass "
                f"force=True to overwrite. This should be rare: this "
                f"ledger's integrity as a real accuracy metric depends on "
                f"not silently rewriting a recorded result."
            )

        entry.actual_outcome = actual_outcome
        entry.actual_outcome_entered_at = datetime.utcnow()
        entry.match = match

        dest = self.ledger_dir / f"{entry_id}.json"
        dest.write_text(entry.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Actual outcome recorded for %s: match=%s", entry_id, match)
        return entry

    def prediction_entries(self, resolved_only: bool = False) -> list[LedgerEntry]:
        """All `entry_type == "prediction"` entries, optionally restricted
        to ones with a real actual outcome already attached
        (`resolved_only=True`) -- the only set `compute_accuracy_report()`
        (and therefore `acheron report --accuracy`) may compute over."""
        entries = [e for e in self.list_entries() if e.entry_type == "prediction"]
        if resolved_only:
            entries = [
                e for e in entries
                if e.actual_outcome is not None and e.match is not None
            ]
        return entries


def compute_accuracy_report(entries: list[LedgerEntry]) -> dict:
    """Compute real predicted-vs-actual accuracy over resolved prediction
    entries ONLY: `entry_type == "prediction"` entries that have since had
    a real `actual_outcome`/`match` attached via `record_actual_outcome()`.

    This is deliberately the ONLY accuracy figure this module can produce.
    There is no backtested/literature-matched variant here to confuse it
    with -- a prediction entry's `timestamp` is set when the prediction was
    made, `actual_outcome_entered_at` when the real result was entered
    afterward, and `record_actual_outcome()` refuses to accept a result for
    anything that isn't a logged prediction. If nothing has been resolved
    yet, this returns `resolved_count=0` and an explicit explanatory note
    rather than substituting some other number that could be mistaken for
    a real accuracy rate.

    The "cited vs UNKNOWN parameters" breakdown the report requires reuses
    `predicted_confidence_tier` (high/medium/low), the same cited-ratio
    bucketing `grn_model.py`/`hypothesis_engine.py` already use everywhere
    else in this codebase, rather than inventing a second threshold here.
    """
    resolved = [
        e for e in entries
        if e.entry_type == "prediction" and e.actual_outcome is not None and e.match is not None
    ]

    if not resolved:
        return {
            "resolved_count": 0,
            "overall": None,
            "by_organism": {},
            "by_confidence_tier": {},
            "note": (
                "No real forward-tested predictions have been resolved yet. "
                "This metric only exists once a prediction is logged BEFORE "
                "a real test (record_prediction) and a real outcome is "
                "entered afterward (record_actual_outcome) -- there is "
                "nothing to report yet, and no other number should be "
                "substituted for it."
            ),
        }

    def _bucket(subset: list[LedgerEntry]) -> dict:
        matches = sum(1 for e in subset if e.match)
        return {
            "n": len(subset),
            "matches": matches,
            "accuracy_pct": round(100.0 * matches / len(subset), 1),
        }

    by_organism: dict[str, list[LedgerEntry]] = {}
    by_tier: dict[str, list[LedgerEntry]] = {}
    for e in resolved:
        by_organism.setdefault(e.organism or "unknown", []).append(e)
        by_tier.setdefault(e.predicted_confidence_tier or "unknown", []).append(e)

    return {
        "resolved_count": len(resolved),
        "overall": _bucket(resolved),
        "by_organism": {k: _bucket(v) for k, v in sorted(by_organism.items())},
        "by_confidence_tier": {k: _bucket(v) for k, v in sorted(by_tier.items())},
    }
