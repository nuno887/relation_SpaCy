# relations.py
from spacy.tokens import Doc

# public API
__all__ = ["build_relations"]

# store edges on the Doc
Doc.set_extension("relations", default=[], force=True)

def _norm(s: str) -> str:
    return " ".join(s.split()).strip().upper().strip(",.;:")

def build_relations(doc: Doc) -> None:
    """
    Builds exactly these edges (no dedupe):
      - ORG -> ORG              (SAME_AS; identical normalized text seen again)
      - ORG -> DOC              (SECTION_ITEM; section-level doc)
      - ORG -> ORG_SECUNDARIA   (CONTAINS; company listed under an ORG section)
      - ORG_SECUNDARIA -> DOC   (HAS_DOCUMENT; proximity-based to latest suborg)
    """
    # helper for SAME_AS
    def _norm(s: str) -> str:
        return " ".join(s.split()).strip().upper().strip(",.;:")

    doc._.relations = []
    ents = sorted(doc.ents, key=lambda e: (e.start_char, -e.end_char))

    current_org = None
    last_suborg = None
    last_org_by_norm = {}
    COMPANY_DOC_CHAR_WINDOW = 300  # tune if needed
    edge_id = 1

    for ent in ents:
        if ent.label_ == "ORG":
            key = _norm(ent.text)
            if key in last_org_by_norm:
                prev = last_org_by_norm[key]
                if prev.start_char != ent.start_char:
                    doc._.relations.append({
                        "id": edge_id,
                        "head": {"text": prev.text, "label": "ORG"},
                        "tail": {"text": ent.text,  "label": "ORG"},
                        "relation": "SAME_AS",
                        "head_offsets": {"start": prev.start_char, "end": prev.end_char},
                        "tail_offsets": {"start": ent.start_char,  "end": ent.end_char},
                        "section_id": prev.start_char,
                        "source_rule": "org_same_norm",
                    })
                    edge_id += 1
            else:
                last_org_by_norm[key] = ent

            current_org = ent
            last_suborg = None

        elif ent.label_ == "ORG_SECUNDARIA":
            if current_org:
                doc._.relations.append({
                    "id": edge_id,
                    "head": {"text": current_org.text, "label": "ORG"},
                    "tail": {"text": ent.text,         "label": "ORG_SECUNDARIA"},
                    "relation": "CONTAINS",
                    "head_offsets": {"start": current_org.start_char, "end": current_org.end_char},
                    "tail_offsets": {"start": ent.start_char,         "end": ent.end_char},
                    "section_id": current_org.start_char,
                    "source_rule": "section_contains_suborg",
                })
                edge_id += 1
            last_suborg = ent

        elif ent.label_ == "DOC":
            if current_org:
                doc._.relations.append({
                    "id": edge_id,
                    "head": {"text": current_org.text, "label": "ORG"},
                    "tail": {"text": ent.text,         "label": "DOC"},
                    "relation": "SECTION_ITEM",
                    "head_offsets": {"start": current_org.start_char, "end": current_org.end_char},
                    "tail_offsets": {"start": ent.start_char,         "end": ent.end_char},
                    "section_id": current_org.start_char,
                    "source_rule": "section_item",
                })
                edge_id += 1

            if last_suborg and (ent.start_char - last_suborg.end_char) <= COMPANY_DOC_CHAR_WINDOW:
                doc._.relations.append({
                    "id": edge_id,
                    "head": {"text": last_suborg.text, "label": "ORG_SECUNDARIA"},
                    "tail": {"text": ent.text,         "label": "DOC"},
                    "relation": "HAS_DOCUMENT",
                    "head_offsets": {"start": last_suborg.start_char, "end": last_suborg.end_char},
                    "tail_offsets": {"start": ent.start_char,         "end": ent.end_char},
                    "section_id": current_org.start_char if current_org else None,
                    "source_rule": "nearest_suborg_window",
                })
                edge_id += 1

