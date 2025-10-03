from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from spacy.tokens import Doc, Span
from models import Sumario, BodyItem

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
def build_sumario_and_body(doc: Doc, include_local_details: bool = False) -> tuple[Sumario, List[BodyItem]]:
    """
    Returns:
      - Sumário block (text + ents + relations) for [0:cut)
      - Body items sliced per ORG → DOC routine you specified
    """
    ents = _ents_in_order(doc)
    cut = _find_body_start_with_first_repeated_org(doc)

    # ----- Sumário -----
    sum_text = doc.text[:cut]
    sum_ents = _filter_ents_in_span(doc, 0, cut)
    sum_rels = _filter_relations_in_span(doc, 0, cut)
    sumario = Sumario(text=sum_text, ents=sum_ents, relations=sum_rels)

    # ----- Body (ORG → DOC only) -----
    # ORGs that start at or after the cut
    body_orgs = [e for e in ents if e.label_ == "ORG" and e.start_char >= cut]
    body_items: List[BodyItem] = []
    order_idx = 1

    # Pre-collect SECTION_ITEM edges grouped by ORG offsets for quick lookup
    org_to_docs: Dict[tuple[int, int], List[Span]] = {}
    for r in _relations_of_type(doc, "SECTION_ITEM"):
        head = (r["head_offsets"]["start"], r["head_offsets"]["end"])
        d_span = _span_by_offsets(doc, r["tail_offsets"]["start"], r["tail_offsets"]["end"], "DOC")
        if d_span:
            org_to_docs.setdefault(head, []).append(d_span)

    for i, org in enumerate(body_orgs):
        # section end: next ORG start or EOF
        next_starts = [o.start_char for o in body_orgs if o.start_char > org.start_char]
        section_end = min(next_starts) if next_starts else len(doc.text)

        # DOCs for this ORG that begin inside the section
        docs = [d for d in org_to_docs.get((org.start_char, org.end_char), []) if org.end_char <= d.start_char < section_end]
        docs.sort(key=lambda d: d.start_char)

        # Slicing routine:
        # - First DOC: [ORG.start : next_DOC.start) or section_end if only one
        # - Middle DOCs: [DOC_i.start : DOC_{i+1}.start)
        # - Last DOC: [DOC_last.start : section_end)
        if not docs:
            continue

        # First DOC
        first = docs[0]
        first_end = docs[1].start_char if len(docs) >= 2 else section_end
        body_items.append(
            _make_body_item(
                doc=doc,
                org=org,
                d=first,
                start=org.start_char,
                end=first_end,
                order_idx=order_idx,
                include_local_details=include_local_details,
            )
        )
        order_idx += 1

        # Middle DOCs
        for j in range(1, len(docs) - 1):
            cur = docs[j]
            nxt = docs[j + 1]
            body_items.append(
                _make_body_item(
                    doc=doc,
                    org=org,
                    d=cur,
                    start=cur.start_char,
                    end=nxt.start_char,
                    order_idx=order_idx,
                    include_local_details=include_local_details,
                )
            )
            order_idx += 1

        # Last DOC
        if len(docs) >= 2:
            last = docs[-1]
            body_items.append(
                _make_body_item(
                    doc=doc,
                    org=org,
                    d=last,
                    start=last.start_char,
                    end=section_end,
                    order_idx=order_idx,
                    include_local_details=include_local_details,
                )
            )
            order_idx += 1

    return sumario, body_items

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
