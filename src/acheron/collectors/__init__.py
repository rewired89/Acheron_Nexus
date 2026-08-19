"""Paper collectors for various academic and bio-database sources."""

from acheron.collectors.arxiv import ArxivCollector
from acheron.collectors.biorxiv import BiorxivCollector
from acheron.collectors.ncbi_gene import NCBIGeneCollector
from acheron.collectors.physionet import PhysioNetCollector
from acheron.collectors.pubmed import PubMedCollector
from acheron.collectors.structures import AlphaFoldCollector, PDBCollector
from acheron.collectors.uniprot import UniProtCollector

__all__ = [
    "PubMedCollector",
    "BiorxivCollector",
    "ArxivCollector",
    "PhysioNetCollector",
    "UniProtCollector",
    "PDBCollector",
    "AlphaFoldCollector",
    "NCBIGeneCollector",
]
