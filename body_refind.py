# body_refind.py
# spaCy-based re-anchoring using PhraseMatcher on the BODY-ONLY doc, no extra normalization

import re
from typing import Dict, List, Tuple
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from models import BodyItem

__all__ = ["build_body_via_sumario_spacy"]

# -------------------- ALL-CAPS gate (for ORG / ORG_SECUNDARIA) --------------------

_caps_token_rx = re.compile(r"[A-Za-zÀ-ÿ]")

def _is_all_caps_token(tok: str) -> bool:
    has_alpha = False
    for ch in tok:
        if ch.isalpha():
            has_alpha = True
            if ch != ch.upper():
                return False
    return has_alpha

def _passes_all_caps_gate(text: str) -> bool:
    # reject spans containing a blank line
    if re.search(r"\n\s*\n", text):
        return False
    for tok in re.split(r"\s+", text.strip()):
        if _caps_token_rx.search(tok) and not _is_all_caps_token(tok):
            return False
    return True

# -------------------- Main --------------------

def build_body_via_sumario_spacy(doc: Doc, roster: Dict[str, object], include_local_details: bool = False) -> List[BodyItem]:
    """
    Match the roster's ORG / ORG_SECUNDARIA / DOC strings in the BODY-ONLY `doc` with spaCy PhraseMatcher,
    enforce ALL-CAPS for ORG/SUBORG, assign in roster order, and slice sections by DOC anchors.
    """
    full_text = doc.text

    # Build simple blueprint directly from roster (strings-only, ordered)
    blueprint = [
        {
            "org_text": o.get("org_text", ""),
            "suborgs": [{"text": t} for t in o.get("suborg_texts", [])],
            "docs":    [{"text": t} for t in o.get("doc_texts",    [])],
        }
        for o in roster.get("orgs", [])
    ]

    # Prepare a PT tokenizer just to segment pattern strings,
    # then rebuild pattern Docs with the SAME vocab as the body doc.
    pt_nlp = spacy.blank("pt")
    def make_pat(text: str) -> Doc:
        tmp = pt_nlp.make_doc(text)
        return Doc(doc.vocab, words=[t.text for t in tmp])

    # One PhraseMatcher, three groups
    matchers: Dict[str, PhraseMatcher] = {
        "ORG": PhraseMatcher(doc.vocab, attr="LOWER"),
        "ORG_SECUNDARIA": PhraseMatcher(doc.vocab, attr="LOWER"),
        "DOC": PhraseMatcher(doc.vocab, attr="LOWER"),
    }
    key_to_phrase: Dict[str, str] = {}

    def add_phrases(label: str, phrases: List[str]) -> None:
        seen = set()
        for i, p in enumerate(phrases):
            if not p or p in seen:
                continue
            seen.add(p)
            pat_doc = make_pat(p)
            key = f"{label}:{i}"
            key_to_phrase[key] = p
            matchers[label].add(key, [pat_doc])

    # Collect phrases from roster in Sumário order
    org_phrases = [b["org_text"] for b in blueprint]
    sub_phrases = [s["text"] for b in blueprint for s in b["suborgs"]]
    doc_phrases = [d["text"] for b in blueprint for d in b["docs"]]

    add_phrases("ORG", org_phrases)
    add_phrases("ORG_SECUNDARIA", sub_phrases)
    add_phrases("DOC", doc_phrases)

    # Run matchers on the BODY doc; collect candidates keyed by the literal phrase text
    def gather_candidates(label: str) -> Dict[str, List[Tuple[int, int]]]:
        out: Dict[str, List[Tuple[int, int]]] = {}
        for match_id, start, end in matchers[label](doc):
            key = doc.vocab.strings[match_id]  # e.g., "ORG:0"
            phrase = key_to_phrase.get(key, "")
            span = doc[start:end]
            if label in ("ORG", "ORG_SECUNDARIA") and not _passes_all_caps_gate(span.text):
                continue
            out.setdefault(phrase, []).append((span.start_char, span.end_char))
        for k in out:
            out[k].sort(key=lambda p: (p[0], -p[1]))
        return out

    org_cands = gather_candidates("ORG")
    sub_cands = gather_candidates("ORG_SECUNDARIA")
    doc_cands = gather_candidates("DOC")

    # Assign in roster order with a moving cursor
    assigned_orgs: List[Dict] = []
    cursor = 0

    def next_org_start(i: int) -> int:
        for j in range(i + 1, len(assigned_orgs)):
            if assigned_orgs[j].get("assigned"):
                return assigned_orgs[j]["assigned"][0]
        return len(full_text)

    # ORGs
    for b in blueprint:
        phrase = b["org_text"]
        cands = org_cands.get(phrase, [])
        chosen = None
        for st, en in cands:
            if st >= cursor:
                chosen = (st, en)
                cursor = en
                break
        assigned_orgs.append({**b, "assigned": chosen})

    # For each ORG section, assign SUBORGs and DOCs; slice sections using DOC anchors
    body_items: List[BodyItem] = []
    order_idx = 1

    for i, org_entry in enumerate(assigned_orgs):
        org_span = org_entry["assigned"]
        if org_span is None:
            continue
        org_st, org_en = org_span
        section_end = next_org_start(i)

        # SUBORGs (optional for slicing; useful if you later emit body relations)
        sub_cursor = org_st
        for sub in org_entry["suborgs"]:
            phrase = sub["text"]
            hits = [h for h in sub_cands.get(phrase, []) if org_st <= h[0] < section_end]
            for st, en in hits:
                if st >= sub_cursor:
                    sub_cursor = en
                    break

        # DOCs drive slicing
        doc_cursor = org_en
        doc_assignments: List[Tuple[int, int]] = []
        for d in org_entry["docs"]:
            phrase = d["text"]
            hits = [h for h in doc_cands.get(phrase, []) if org_en <= h[0] < section_end]
            chosen = None
            for st, en in hits:
                if st >= doc_cursor:
                    chosen = (st, en)
                    doc_cursor = en
                    break
            if chosen:
                doc_assignments.append(chosen)

        if not doc_assignments:
            continue

        # First slice: [ORG.start : first DOC.start)
        first_doc_st, first_doc_en = doc_assignments[0]
        first_end = (doc_assignments[1][0] if len(doc_assignments) >= 2 else section_end)
        body_items.append(BodyItem(
            org_text=" ".join(org_entry["org_text"].split()),
            org_start=org_st,
            org_end=org_en,
            section_id=org_st,
            doc_title=full_text[first_doc_st:first_doc_en],
            doc_start=first_doc_st,
            doc_end=first_doc_en,
            relation="SECTION_ITEM",
            slice_text=full_text[org_st:first_end].strip(),
            slice_start=org_st,
            slice_end=first_end,
            order_index=order_idx,
        ))
        order_idx += 1

        # Middles: [DOC_i.start : DOC_{i+1}.start)
        for j in range(1, len(doc_assignments) - 1):
            cur_st, cur_en = doc_assignments[j]
            nxt_st, _ = doc_assignments[j + 1]
            body_items.append(BodyItem(
                org_text=" ".join(org_entry["org_text"].split()),
                org_start=org_st,
                org_end=org_en,
                section_id=org_st,
                doc_title=full_text[cur_st:cur_en],
                doc_start=cur_st,
                doc_end=cur_en,
                relation="SECTION_ITEM",
                slice_text=full_text[cur_st:nxt_st].strip(),
                slice_start=cur_st,
                slice_end=nxt_st,
                order_index=order_idx,
            ))
            order_idx += 1

        # Last: [last DOC.start : section_end)
        if len(doc_assignments) >= 2:
            last_st, last_en = doc_assignments[-1]
            body_items.append(BodyItem(
                org_text=" ".join(org_entry["org_text"].split()),
                org_start=org_st,
                org_end=org_en,
                section_id=org_st,
                doc_title=full_text[last_st:last_en],
                doc_start=last_st,
                doc_end=last_en,
                relation="SECTION_ITEM",
                slice_text=full_text[last_st:section_end].strip(),
                slice_start=last_st,
                slice_end=section_end,
                order_index=order_idx,
            ))
            order_idx += 1

    return body_items
