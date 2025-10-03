# body_refind.py
# spaCy-based re-anchoring of Sumário entities into the body (with ALL-CAPS gate)
import re
import unicodedata
from typing import Dict, List, Tuple, Optional

import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from models import BodyItem

__all__ = ["build_body_via_sumario_spacy"]

# -------------------- Normalization helpers --------------------

def _strip_diacritics(s: str) -> str:
    # NFKD then drop combining marks
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def _normalize_for_match(s: str) -> str:
    """
    Canonical form (applied to Sumário strings):
      - remove diacritics
      - uppercase
      - collapse whitespace (incl. newlines) to a single space
      - join soft hyphenations (letter '-' + whitespace* + letter)
      - remove dots inside ALL-CAPS acronyms: 'E.P.E.' -> 'EPE'
      - trim light edge punctuation
    """
    s = _strip_diacritics(s)
    s = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", s)           # soft hyphenation
    s = re.sub(r"(?<=\b[A-Z])\.(?=[A-Z]\b)", "", s)               # E.P.E. -> EPE
    s = re.sub(r"\s+", " ", s).strip()
    s = s.upper().strip(" ,.;:—–-")
    return s

def _build_normalized_body_with_map(text: str) -> Tuple[str, List[int]]:
    """
    Build normalized body string (uppercase, diacritics stripped, whitespace collapsed,
    soft hyphenation fixed, acronym dots removed) AND a char->original index map.
    We do NOT strip leading/trailing punctuation here to keep mapping simple.
    """
    out_chars: List[str] = []
    idx_map: List[int] = []
    i = 0
    last_was_space = False
    while i < len(text):
        ch = text[i]
        # Join soft hyphenations: letter '-' ws* letter  => skip '-' + ws
        if ch == "-" and i > 0 and text[i - 1].isalpha():
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j].isalpha():
                i = j
                continue
        if ch.isspace():
            if not last_was_space:
                out_chars.append(" ")
                idx_map.append(i)
                last_was_space = True
            i += 1
            continue
        last_was_space = False
        # Uppercase + strip diacritics per char
        ch_norm = _strip_diacritics(ch).upper()
        # Drop dots inside ALL-CAPS acronyms
        if ch_norm == "." and out_chars:
            prev = out_chars[-1]
            nxt = text[i + 1].upper() if i + 1 < len(text) else ""
            if prev.isalpha() and prev == prev.upper() and nxt.isalpha() and nxt == nxt.upper():
                i += 1
                continue
        out_chars.append(ch_norm)
        idx_map.append(i)
        i += 1
    return "".join(out_chars), idx_map

# -------------------- Gates & utilities --------------------

_caps_token_rx = re.compile(r"[A-Za-zÀ-ÿ]")

def _is_all_caps_token(tok: str) -> bool:
    has_alpha = False
    for ch in tok:
        if ch.isalpha():
            has_alpha = True
            if ch != ch.upper():
                return False
    return has_alpha

def _passes_all_caps_gate(original_span: str) -> bool:
    # Must be contiguous (no blank line inside)
    if re.search(r"\n\s*\n", original_span):
        return False
    # Every token with a letter must be ALL CAPS (Unicode-aware)
    for tok in re.split(r"\s+", original_span.strip()):
        if _caps_token_rx.search(tok) and not _is_all_caps_token(tok):
            return False
    return True

def _word_boundary_ok(norm_text: str, start: int, end: int) -> bool:
    left_ok = start == 0 or not norm_text[start - 1].isalnum()
    right_ok = end == len(norm_text) or not norm_text[end].isalnum()
    return left_ok and right_ok

def _sorted_by_start(items: List[Tuple[int,int,str,str]]) -> List[Tuple[int,int,str,str]]:
    return sorted(items, key=lambda t: (t[0], -t[1]))

def _build_blueprint(sumario) -> List[dict]:
    """
    Ordered Sumário blueprint:
      [
        {
          "org_text": str,
          "org_offsets": (start,end),
          "suborgs": [ {"text": str, "offsets": (s,e)}, ... ],
          "docs": [ {"text": str, "offsets": (s,e)}, ... ],
        }, ...
      ]
    Order is exactly the Sumário ORG occurrence order.
    """
    ents = sumario.ents  # Dict[str, List[(start,end,label,text)]]
    orgs = _sorted_by_start(ents.get("ORG", []))
    # quick lookup for text by offsets
    text_by_offsets = {(st, en, lab): txt for lab, lst in ents.items() for (st, en, lab, txt) in lst}

    rels = sumario.relations or []
    org_to_sub: Dict[Tuple[int,int], List[Tuple[int,int]]] = {}
    org_to_doc: Dict[Tuple[int,int], List[Tuple[int,int]]] = {}

    for r in rels:
        rel = r.get("relation")
        hs, he = r["head_offsets"]["start"], r["head_offsets"]["end"]
        ts, te = r["tail_offsets"]["start"], r["tail_offsets"]["end"]
        if rel == "CONTAINS":        # ORG -> ORG_SECUNDARIA
            org_to_sub.setdefault((hs, he), []).append((ts, te))
        elif rel == "SECTION_ITEM":  # ORG -> DOC
            org_to_doc.setdefault((hs, he), []).append((ts, te))

    for k in org_to_sub:
        org_to_sub[k].sort(key=lambda p: p[0])
    for k in org_to_doc:
        org_to_doc[k].sort(key=lambda p: p[0])

    blueprint: List[dict] = []
    for (st, en, _, org_text) in orgs:
        key = (st, en)
        suborgs = [{"text": text_by_offsets.get((ts, te, "ORG_SECUNDARIA"), ""), "offsets": (ts, te)}
                   for (ts, te) in org_to_sub.get(key, [])]
        docs = [{"text": text_by_offsets.get((ts, te, "DOC"), ""), "offsets": (ts, te)}
                for (ts, te) in org_to_doc.get(key, [])]
        blueprint.append({
            "org_text": org_text,
            "org_offsets": (st, en),
            "suborgs": suborgs,
            "docs": docs,
        })
    return blueprint

# -------------------- Pass 2: spaCy PhraseMatcher --------------------

def build_body_via_sumario_spacy(doc: Doc, sumario, include_local_details: bool = False) -> List[BodyItem]:
    """
    Re-find the exact (normalized) ORG / ORG_SECUNDARIA / DOC strings from the Sumário
    in the body using spaCy's PhraseMatcher, preserving Sumário order.
    ORG/ORG_SECUNDARIA candidates must pass an ALL-CAPS gate on the ORIGINAL text
    (multi-line headers allowed).
    """
    full_text = doc.text
    cut = len(sumario.text)   # body starts here
    body_text = full_text[cut:]

    blueprint = _build_blueprint(sumario)

    # Build normalized body + map back to original offsets
    norm_body, idx_map = _build_normalized_body_with_map(body_text)

    # Tiny spaCy pipeline on normalized body
    nlp = spacy.blank("pt")
    norm_doc = nlp.make_doc(norm_body)

    # Prepare phrase matchers per label, indexed by normalized phrase
    matchers: Dict[str, PhraseMatcher] = {
        "ORG": PhraseMatcher(nlp.vocab, attr="LOWER"),
        "ORG_SECUNDARIA": PhraseMatcher(nlp.vocab, attr="LOWER"),
        "DOC": PhraseMatcher(nlp.vocab, attr="LOWER"),
    }
    key_to_norm: Dict[str, str] = {}

    def add_phrases(label: str, phrases: List[str]) -> None:
        seen_norm = set()
        for i, p in enumerate(phrases):
            norm = _normalize_for_match(p)
            if not norm or norm in seen_norm:
                continue
            seen_norm.add(norm)
            pat_doc = nlp.make_doc(norm)
            key = f"{label}:{i}"
            key_to_norm[key] = norm
            matchers[label].add(key, [pat_doc])

    # Collect phrases in Sumário order
    org_phrases = [b["org_text"] for b in blueprint]
    sub_phrases = [s["text"] for b in blueprint for s in b["suborgs"]]
    doc_phrases = [d["text"] for b in blueprint for d in b["docs"]]

    add_phrases("ORG", org_phrases)
    add_phrases("ORG_SECUNDARIA", sub_phrases)
    add_phrases("DOC", doc_phrases)

    # Run matchers on the normalized body and collect candidates per normalized phrase
    def gather_candidates(label: str) -> Dict[str, List[Tuple[int, int]]]:
        out: Dict[str, List[Tuple[int, int]]] = {}
        for match_id, start, end in matchers[label](norm_doc):
            key = nlp.vocab.strings[match_id]
            norm_phrase = key_to_norm.get(key)
            if not norm_phrase:
                continue
            # char span in normalized body (tokens give char indices)
            norm_start = norm_doc[start].idx
            last_tok = norm_doc[end - 1]
            norm_end = last_tok.idx + len(last_tok.text)
            # word-boundary guard
            if not _word_boundary_ok(norm_body, norm_start, norm_end):
                continue
            # map back to original full-text offsets
            orig_start = idx_map[norm_start]
            orig_end = idx_map[norm_end - 1] + 1
            full_start = cut + orig_start
            full_end = cut + orig_end
            # ALL-CAPS gate for ORG/SUBORG on original text
            if label in ("ORG", "ORG_SECUNDARIA"):
                if not _passes_all_caps_gate(full_text[full_start:full_end]):
                    continue
            out.setdefault(norm_phrase, []).append((full_start, full_end))
        # sort each candidate list by start (stable)
        for k in out:
            out[k].sort(key=lambda p: (p[0], -p[1]))
        return out

    org_cands = gather_candidates("ORG")
    sub_cands = gather_candidates("ORG_SECUNDARIA")
    doc_cands = gather_candidates("DOC")

    # -------------------- Monotone assignment (Sumário order) --------------------

    assigned_orgs: List[Dict] = []
    cursor = cut  # global cursor through the body

    def next_org_start(i: int) -> int:
        for j in range(i + 1, len(assigned_orgs)):
            if assigned_orgs[j].get("assigned"):
                return assigned_orgs[j]["assigned"][0]
        return len(full_text)

    # Assign ORGs in Sumário order
    for b in blueprint:
        norm = _normalize_for_match(b["org_text"])
        cands = org_cands.get(norm, [])
        chosen = None
        for st, en in cands:
            if st >= cursor:
                chosen = (st, en)
                cursor = en
                break
        assigned_orgs.append({**b, "assigned": chosen})

    # For each ORG section, assign SUBORGs and DOCs in-order within window
    body_items: List[BodyItem] = []
    order_idx = 1

    for i, org_entry in enumerate(assigned_orgs):
        org_span = org_entry["assigned"]
        if org_span is None:
            continue  # cannot make a section without a body anchor
        org_st, org_en = org_span
        section_end = next_org_start(i)

        # SUBORGs (assignment only; slicing uses DOCs)
        sub_cursor = org_st
        for sub in org_entry["suborgs"]:
            norm = _normalize_for_match(sub["text"])
            hits = [h for h in sub_cands.get(norm, []) if org_st <= h[0] < section_end]
            chosen = None
            for st, en in hits:
                if st >= sub_cursor:
                    chosen = (st, en)
                    sub_cursor = en
                    break
            # (We don't need to store sub assignments for slicing, but they’re
            # useful if later you want to emit ORG->ORG_SECUNDARIA body relations.)

        # DOCs (drive slicing)
        doc_cursor = org_en  # docs must follow the org header
        doc_assignments: List[Tuple[int, int]] = []
        for d in org_entry["docs"]:
            norm = _normalize_for_match(d["text"])
            hits = [h for h in doc_cands.get(norm, []) if org_en <= h[0] < section_end]
            chosen = None
            for st, en in hits:
                if st >= doc_cursor:
                    chosen = (st, en)
                    doc_cursor = en
                    break
            if chosen:
                doc_assignments.append(chosen)

        if not doc_assignments:
            continue  # nothing to slice under this ORG

        # Build slices like your segmenter: first, middle(s), last
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
