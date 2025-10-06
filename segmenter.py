from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from spacy.tokens import Doc, Span
from models import Sumario

# -------- helpers --------
def _collapse_ws(s: str) -> str:
    return " ".join(s.split())

def _ents_in_order(doc: Doc) -> List[Span]:
    return sorted(doc.ents, key=lambda e: (e.start_char, -e.end_char))

def _next_entity_start_after(doc: Doc, pos: int) -> int:
    nxt = [e.start_char for e in doc.ents if e.start_char > pos]
    return min(nxt) if nxt else len(doc.text)

def _relations_of_type(doc: Doc, rel_label: str) -> List[dict]:
    return [r for r in getattr(doc._, "relations", []) if r.get("relation") == rel_label]

def _span_by_offsets(doc: Doc, start: int, end: int, label: Optional[str] = None) -> Optional[Span]:
    for e in doc.ents:
        if e.start_char == start and e.end_char == end and (label is None or e.label_ == label):
            return e
    return None

def _norm_org(s: str) -> str:
    return _collapse_ws(s).upper().strip(",.;:")

def _filter_ents_in_span(doc: Doc, a: int, b: int) -> Dict[str, List[Tuple[int, int, str, str]]]:
    out: Dict[str, List[Tuple[int, int, str, str]]] = {"ORG": [], "DOC": [], "ORG_SECUNDARIA": []}
    for e in doc.ents:
        if a <= e.start_char and e.end_char <= b and e.label_ in out:
            out[e.label_].append((e.start_char, e.end_char, e.label_, e.text))
    for k in out:
        out[k].sort(key=lambda t: (t[0], -t[1]))
    return out

def _filter_relations_in_span(doc: Doc, a: int, b: int) -> List[dict]:
    rels = []
    for r in getattr(doc._, "relations", []):
        hs, he = r["head_offsets"]["start"], r["head_offsets"]["end"]
        ts, te = r["tail_offsets"]["start"], r["tail_offsets"]["end"]
        if a <= hs < b and a <= ts < b:
            rels.append(r)
    return rels

def _find_body_start_with_first_repeated_org(doc: Doc) -> int:
    """Second occurrence (by normalized text) of an ORG marks the body start."""
    seen: Dict[str, int] = {}
    for e in _ents_in_order(doc):
        if e.label_ != "ORG":
            continue
        key = _norm_org(e.text)
        if key in seen:
            return e.start_char
        seen[key] = e.start_char
    # If no repeat, assume no Sumário; but per your note we can just return len(doc) to keep summary empty.
    return len(doc.text)

# -------- main API --------
def build_sumario_and_body(doc: Doc, include_local_details: bool = False) -> Tuple[Sumario, Dict[str, object], str]:
    """
    Returns:
      - sumario: Sumário block (text + ents + relations) for [0:cut)
      - roster:  minimal, strings-only hand-off for Pass 2:
                 {"cut_index": int, "orgs": [{"org_text": str,
                                              "suborg_texts": [str, ...],
                                              "doc_texts": [str, ...]}, ...]}
      - body_text: original text from cut_index to EOF
    """
    # --- Sumário (unchanged) ---
    ents = _ents_in_order(doc)
    cut = _find_body_start_with_first_repeated_org(doc)

    sum_text = doc.text[:cut]
    sum_ents = _filter_ents_in_span(doc, 0, cut)
    sum_rels = _filter_relations_in_span(doc, 0, cut)
    sumario = Sumario(text=sum_text, ents=sum_ents, relations=sum_rels)

    # --- Roster (strings-only, ordered) ---
    cut_index = len(sumario.text)
    ent_text_by_offsets = {
        (st, en, lab): txt
        for lab, lst in sum_ents.items()
        for (st, en, lab, txt) in lst
    }

    # Group Sumário relations by ORG offsets
    org_to_sub: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    org_to_doc: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for r in sum_rels:
        rel = r.get("relation")
        hs, he = r["head_offsets"]["start"], r["head_offsets"]["end"]
        ts, te = r["tail_offsets"]["start"], r["tail_offsets"]["end"]
        if rel == "CONTAINS":
            org_to_sub.setdefault((hs, he), []).append((ts, te))
        elif rel == "SECTION_ITEM":
            org_to_doc.setdefault((hs, he), []).append((ts, te))

    for k in org_to_sub:
        org_to_sub[k].sort(key=lambda p: p[0])
    for k in org_to_doc:
        org_to_doc[k].sort(key=lambda p: p[0])

    # ORG occurrences in Sumário order (use offsets only for sorting; not returned)
    orgs_ordered = sorted(sum_ents.get("ORG", []), key=lambda t: (t[0], -t[1]))

    roster_orgs: List[Dict[str, object]] = []
    for st, en, _, org_text in orgs_ordered:
        key = (st, en)
        sub_texts = [
            ent_text_by_offsets.get((ts, te, "ORG_SECUNDARIA"), "")
            for (ts, te) in org_to_sub.get(key, [])
        ]
        doc_texts = [
            ent_text_by_offsets.get((ts, te, "DOC"), "")
            for (ts, te) in org_to_doc.get(key, [])
        ]
        roster_orgs.append(
            {"org_text": _collapse_ws(org_text),
             "suborg_texts":[_collapse_ws(t) for t in sub_texts], 
             "doc_texts": [_collapse_ws(t) for t in doc_texts],
             }
        )

    roster: Dict[str, object] = {"cut_index": cut_index, "orgs": roster_orgs}
    body_text = doc.text[cut_index:]
    return sumario, roster, body_text



"""
def _make_body_item(
    doc: Doc,
    org: Span,
    d: Span,
    start: int,
    end: int,
    order_idx: int,
    include_local_details: bool,
) -> BodyItem:
    slice_text = doc.text[start:end].strip()
    item = BodyItem(
        org_text=_collapse_ws(org.text),
        org_start=org.start_char,
        org_end=org.end_char,
        section_id=org.start_char,
        doc_title=d.text,
        doc_start=d.start_char,
        doc_end=d.end_char,
        relation="SECTION_ITEM",
        slice_text=slice_text,
        slice_start=start,
        slice_end=end,
        order_index=order_idx,
    )
    if include_local_details:
        item.ents_in_slice = _filter_ents_in_span(doc, start, end)
        item.relations_in_slice = _filter_relations_in_span(doc, start, end)
    return item
"""