from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional  # <-- use Optional, no PEP 604 unions

@dataclass
class Sumario:
    text: str
    ents: Dict[str, List[Tuple[int, int, str, str]]]  # [(start, end, label, text), ...]
    relations: List[dict]                              # copy of doc._.relations filtered to region

@dataclass
class BodyItem:
    # ORG context
    org_text: str
    org_start: int
    org_end: int
    section_id: int

    # DOC slice
    doc_title: str
    doc_start: int
    doc_end: int
    relation: str  # "SECTION_ITEM" (for this phase)

    # Extracted slice text
    slice_text: str
    slice_start: int
    slice_end: int

    # Ordering
    order_index: int

    # (optional) local entities/relations inside the slice (handy for exports/debug)
    ents_in_slice: Optional[Dict[str, List[Tuple[int, int, str, str]]]] = None
    relations_in_slice: Optional[List[dict]] = None
