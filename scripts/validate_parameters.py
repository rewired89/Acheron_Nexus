#!/usr/bin/env python3
"""Validate the extracted parameter corpus: how much is a real cited number
vs. an explicit UNKNOWN, broken down per organism.

This is the number Phase 3 was built to surface, not hide: the RAG corpus
answers questions in prose, but a downstream simulation needs discrete
rate/affinity values. This script reports, per organism, what fraction of
extracted (subject)-[relationship]->(object) records actually carry a cited
numeric rate/affinity value versus how many are honestly UNKNOWN, so nobody
has to guess how simulation-ready the current corpus is.

Also reports the confidence-tier breakdown per organism. science_filter's
_ORGANISM_TIERS includes a "bacteria" tier (added alongside this script) so
SubtiWiki/B. subtilis records are scored against their own organism instead
of being silently penalized for not matching a planarian-only scorer -- see
extraction/parameter_extractor.py's module docstring for how that mapping
works.

Usage:
    python scripts/validate_parameters.py
    python scripts/validate_parameters.py --vectorstore-dir data/vectorstore

Windows PowerShell:
    python scripts\\validate_parameters.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the extracted parameter corpus")
    parser.add_argument(
        "--vectorstore-dir", type=Path, default=None,
        help="ChromaDB persist dir to read chunks from (default: the configured vectorstore_dir)",
    )
    parser.add_argument(
        "--params-dir", type=Path, default=None,
        help="Where to persist extracted ParameterRecords (default: data_dir/parameters)",
    )
    args = parser.parse_args()

    from acheron.extraction.parameter_extractor import ConfidenceTier, UNKNOWN, extract_from_store
    from acheron.vectorstore.store import VectorStore

    store = VectorStore(persist_dir=args.vectorstore_dir)
    total_chunks = store.count()
    if total_chunks == 0:
        print("No chunks indexed yet -- run `acheron collect` then `acheron index` first.")
        return

    records = extract_from_store(store, params_dir=args.params_dir)
    if not records:
        print(f"Read {total_chunks} indexed chunks but extracted 0 parameter records.")
        return

    by_organism: dict[str, list] = defaultdict(list)
    for r in records:
        by_organism[r.organism or "(unspecified)"].append(r)

    print(f"Indexed chunks: {total_chunks}")
    print(f"Extracted parameter records: {len(records)}")
    print()
    print(f"{'Organism':<20} {'Records':>8} {'Cited rate/affinity':>20} {'UNKNOWN':>10} {'% cited':>8}")
    print("-" * 70)

    for organism in sorted(by_organism):
        recs = by_organism[organism]
        cited = sum(1 for r in recs if r.rate_or_affinity != UNKNOWN)
        unknown = len(recs) - cited
        pct = (cited / len(recs)) * 100 if recs else 0.0
        print(f"{organism:<20} {len(recs):>8} {cited:>20} {unknown:>10} {pct:>7.1f}%")

    print()
    print(f"{'Organism':<20} {'HIGH':>6} {'MEDIUM':>8} {'LOW':>6}")
    print("-" * 44)
    for organism in sorted(by_organism):
        recs = by_organism[organism]
        tiers = defaultdict(int)
        for r in recs:
            tiers[r.confidence_tier] += 1
        print(
            f"{organism:<20} {tiers[ConfidenceTier.HIGH]:>6} "
            f"{tiers[ConfidenceTier.MEDIUM]:>8} {tiers[ConfidenceTier.LOW]:>6}"
        )

    heuristic_count = sum(1 for r in records if r.heuristic)
    if heuristic_count:
        print(
            f"\n{heuristic_count} of {len(records)} records came from the heuristic "
            "literature fallback parser (not SubtiWiki/PlanMine's own structured "
            "text) -- treat these as lower-confidence than the structured-source records."
        )

    unscored_bacterial = [
        r for r in records
        if r.source == "subtiwiki" and r.confidence_breakdown.get("organism_match") == 0.0
    ]
    if unscored_bacterial:
        print(
            f"\nNOTE: {len(unscored_bacterial)} SubtiWiki-derived record(s) still scored "
            "organism_match=0.0 despite the bacteria tier -- their chunk's organism field "
            "did not read as 'B. subtilis' (check _ORGANISM_TARGET_MAP in "
            "extraction/parameter_extractor.py if this looks wrong)."
        )


if __name__ == "__main__":
    main()
