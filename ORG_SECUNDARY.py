# gazette_pipeline_structural.py
# Structural-first extractor for gazette-like texts (Portuguese):
# - ORG headers: multi-line, started by "starter" words; continuation lines use grammar-based cues
# - ORG_SECUNDARIA: first post-header lines with enough content or followed by company-level DOCs
# - DOC labels: section-level & company-level (e.g., "Contrato de sociedade")
# - Relations:
#     ORG --SECTION_ITEM--> DOC
#     ORG --CONTAINS--> ORG_SECUNDARIA
#     ORG_SECUNDARIA --HAS_DOCUMENT--> DOC
#
# Usage:
#   python gazette_pipeline_structural.py input.txt
#   cat input.txt | python gazette_pipeline_structural.py

import sys
import re
import unicodedata
from typing import List, Tuple, Optional
import spacy
from spacy.tokens import Doc, Span


# ---------------- Config ----------------
HEADER_STARTERS = {
    "SECRETARIA", "SECRETARIAS", "VICE-PRESIDÊNCIA", "VICE-PRESIDENCIA",
    "PRESIDÊNCIA", "PRESIDENCIA", "DIREÇÃO", "DIRECÇÃO",
    "ASSEMBLEIA", "CÂMARA", "CAMARA", "MUNICIPIO",
    "TRIBUNAL", "CONSERVATÓRIA", "CONSERVATORIA",
    "PRESIDÊNCIA DO GOVERNO", "PRESIDENCIA DO GOVERNO", "APRAM"
}

# Function words to ignore when counting "content tokens"
STOPWORDS_UP = {"DO", "DA", "DE", "DOS", "DAS", "E", "A", "O", "EM", "PARA", "COM", "NO", "NA", "NOS", "NAS"}

# DOC labels (line-start)
DOC_LABELS_SECTION = {
    "RETIFICAÇÃO", "RECTIFICAÇÃO", "RETIFICACAO", "RECTIFICACAO",
    "AVISO", "AVISOS",
    "DESPACHO", "DESPACHO CONJUNTO",
    "EDITAL", "DELIBERAÇÃO", "DELIBERACAO",
    "DECLARAÇÃO", "DECLARACAO",
    "LISTA", "LISTAS",
    "ANÚNCIO", "ANUNCIO", "ANÚNCIO (RESUMO)", "ANUNCIO (RESUMO)",
    "CONVOCATÓRIA", "CONVOCATORIA"
}

# Company-level doc anchor (used for look-ahead)
RX_CONTRATO_SOC = re.compile(r"(?is)\bcontrato\s*de\s*sociedade\b")

# Optional “institutional” starters for secondary orgs (used inside sections)
SECONDARY_STARTERS = {"INSTITUTO", "ASSOCIAÇÃO", "ASSOCIACAO", "CLUBE", "FUNDAÇÃO", "FUNDACAO", "DIREÇÃO", "DIRECÇÃO"}


# ---------------- Normalization & helpers ----------------
def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\ufeff", "").replace("\u00ad", "").replace("\u200b", "")
    s = s.replace("\u00a0", " ")
    s = s.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("…", "...")
    return s

def line_offsets(text: str) -> List[Tuple[int, int, str]]:
    out, i = [], 0
    for ln in text.splitlines(keepends=True):
        out.append((i, i + len(ln), ln))
        i += len(ln)
    return out

def strip(line: str) -> str:
    return line.strip()

def first_alpha_word_upper(line: str) -> str:
    for w in strip(line).split():
        if any(ch.isalpha() for ch in w):
            return unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode().upper().strip(",.;:-")
    return ""

def starts_with_header_starter(line: str) -> bool:
    up = strip(line).upper()
    if not up:
        return False
    first = first_alpha_word_upper(up)
    if first in HEADER_STARTERS:
        return True
    # also allow presence of multiword starters inside the line start (e.g., "PRESIDÊNCIA DO GOVERNO")
    return any(up.startswith(s) for s in HEADER_STARTERS)

def is_blank(line: str) -> bool:
    return strip(line) == ""

def is_doc_label_line(line: str) -> bool:
    up = strip(line).upper()
    if not up:
        return False
    head = " ".join(up.split())  # collapse spaces
    # exact label hits
    if head in DOC_LABELS_SECTION:
        return True
    # numbered forms like "DESPACHO n.º 59/2012"
    if head.startswith("DESPACHO") or head.startswith("DECLARAÇÃO") or head.startswith("DECLARACAO") \
       or head.startswith("RETIFICAÇÃO") or head.startswith("RECTIFICAÇÃO") or head.startswith("AVISO") \
       or head.startswith("AVISOS") or head.startswith("EDITAL") or head.startswith("ANÚNCIO") \
       or head.startswith("ANUNCIO"):
        if any(nm in head for nm in ("N.º", "Nº", "N°", "N.O", "N.O.")):
            return True
    # contrato de sociedade
    if RX_CONTRATO_SOC.search(up):
        return True
    return False

def content_token_count(line: str) -> int:
    toks = [t for t in strip(line).split() if any(ch.isalpha() for ch in t)]
    toks_up = [unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().upper().strip(",.;:") for t in toks]
    return sum(1 for t in toks_up if t not in STOPWORDS_UP)

def is_header_continuation(prev_line: str, curr_line: str) -> bool:
    """Continuation cues: current starts with stopword and has content nouns; or prev ends with connector/comma/hyphen."""
    if is_blank(curr_line):
        return False
    curr_up = strip(curr_line).upper()
    # starts with a function word and has content after
    parts = curr_up.split()
    if parts and parts[0] in STOPWORDS_UP and content_token_count(curr_up) >= 1:
        return True
    # previous ends with a joiner/comma/hyphen
    prev_up = strip(prev_line).upper()
    if prev_up.endswith((" E", " DO", " DA", " DE", " DOS", " DAS")):
        return True
    if prev_up.endswith((",", "-", "–")):
        return True
    # domain nouns after a stopword (e.g., "DO PLANO E FINANÇAS")
    if parts and parts[0] in STOPWORDS_UP and any(w in curr_up for w in ("PLANO", "FINAN", "CULTURA", "TURISMO", "TRANSPORT", "AMBIENTE", "RECURSOS")):
        return True
    return False

def looks_like_secondary_start(line: str) -> bool:
    """Secondary orgs often start with institutional nouns inside a section."""
    up = strip(line).upper()
    if not up:
        return False
    first = first_alpha_word_upper(up)
    if first in SECONDARY_STARTERS:
        return True
    return False


# ---------------- Span builders ----------------
def char_span(doc: Doc, start: int, end: int, label: str) -> Optional[Span]:
    sp = doc.char_span(start, end, label=label, alignment_mode="expand")
    return sp if sp and sp.text.strip() else None


# ---------------- Detection (structural, line-based) ----------------
def detect_entities(doc: Doc) -> List[Span]:
    """State machine over lines to produce ORG, ORG_SECUNDARIA, DOC spans with robust boundaries."""
    spans: List[Span] = []
    lines = line_offsets(doc.text)

    i = 0
    state = "OUTSIDE"
    # temp header block boundaries
    header_start = None
    header_end = None
    max_header_lines = 3  # safe cap

    def close_header(at_index: int):
        nonlocal header_start, header_end, spans
        if header_start is not None and header_end is not None and header_end > header_start:
            sp = char_span(doc, header_start, header_end, "ORG")
            if sp:
                spans.append(sp)
        header_start = header_end = None

    while i < len(lines):
        start, end, ln = lines[i]
        s = strip(ln)

        if state == "OUTSIDE":
            if is_blank(ln):
                i += 1
                continue
            # New header?
            if starts_with_header_starter(ln):
                state = "IN_ORG"
                header_start = start
                header_end = end
                header_lines = 1
                # try to absorb up to max_header_lines with continuation cues
                j = i + 1
                while j < len(lines) and header_lines < max_header_lines:
                    _, end_j, ln_j = lines[j]
                    if is_blank(ln_j) or is_doc_label_line(ln_j) or starts_with_header_starter(ln_j):
                        break
                    if is_header_continuation(ln, ln_j):
                        header_end = end_j
                        ln = ln_j  # update prev for next continuation check
                        header_lines += 1
                        j += 1
                    else:
                        break
                i = j
                # header closed; emit and switch to section
                close_header(i)
                state = "IN_SECTION"
                continue
            else:
                # not a header, maybe DOC at top?
                if is_doc_label_line(ln):
                    sp = char_span(doc, start, end, "DOC")
                    if sp:
                        spans.append(sp)
                i += 1
                continue

        if state == "IN_SECTION":
            if i >= len(lines):
                break
            # Check for start of a new ORG header
            if not is_blank(ln) and starts_with_header_starter(ln):
                state = "IN_ORG"
                header_start = start
                header_end = end
                header_lines = 1
                j = i + 1
                while j < len(lines) and header_lines < max_header_lines:
                    _, end_j, ln_j = lines[j]
                    if is_blank(ln_j) or is_doc_label_line(ln_j) or starts_with_header_starter(ln_j):
                        break
                    if is_header_continuation(ln, ln_j):
                        header_end = end_j
                        ln = ln_j
                        header_lines += 1
                        j += 1
                    else:
                        break
                i = j
                close_header(i)
                state = "IN_SECTION"
                continue

            # If DOC line
            if is_doc_label_line(ln):
                sp = char_span(doc, start, end, "DOC")
                if sp:
                    spans.append(sp)
                i += 1
                continue

            # Secondary org decision:
            # Rule A: content tokens (ignoring stopwords) >= 4 and not header/doc
            # Rule B: look-ahead 1–2 lines finds "Contrato de sociedade"
            promote_secondary = False

            if not is_blank(ln):
                if content_token_count(ln) >= 4 and not starts_with_header_starter(ln):
                    promote_secondary = True
                else:
                    # look-ahead for contrato de sociedade
                    lookahead_window = 2
                    la = 1
                    while la <= lookahead_window and (i + la) < len(lines):
                        la_line = strip(lines[i + la][2]).upper()
                        if RX_CONTRATO_SOC.search(la_line):
                            promote_secondary = True
                            break
                        # stop lookahead at the next header/doc
                        if starts_with_header_starter(la_line) or is_doc_label_line(la_line):
                            break
                        la += 1

            if promote_secondary:
                sp = char_span(doc, start, end, "ORG_SECUNDARIA")
                if sp:
                    # Merge with one continuation line if not header/doc and non-empty
                    if (i + 1) < len(lines):
                        n_start, n_end, n_ln = lines[i + 1]
                        if not is_blank(n_ln) and not starts_with_header_starter(n_ln) and not is_doc_label_line(n_ln):
                            # simple continuation join (common wrapped names)
                            sp2 = char_span(doc, start, n_end, "ORG_SECUNDARIA")
                            if sp2:
                                spans.append(sp2)
                                i += 2
                                continue
                    spans.append(sp)
                i += 1
                continue

            # Otherwise, not header, not doc, not secondary → skip (plain text)
            i += 1
            continue

    return spacy.util.filter_spans(spans)


# ---------------- Relations ----------------
Doc.set_extension("relations", default=[], force=True)

def is_company_doc_label_text(s: str) -> bool:
    up = " ".join(s.upper().split())
    if up in {"CONTRATO DE SOCIEDADE"}:
        return True
    return RX_CONTRATO_SOC.search(up) is not None

def build_relations(doc: Doc) -> None:
    doc._.relations = []
    ents = spacy.util.filter_spans(sorted(list(doc.ents), key=lambda e: (e.start_char, -e.end_char)))
    current_org = None
    current_sub = None

    for ent in ents:
        if ent.label_ == "ORG":
            current_org = ent
            current_sub = None

        elif ent.label_ == "ORG_SECUNDARIA":
            if current_org:
                doc._.relations.append({
                    "head": {"text": current_org.text, "label": "ORG"},
                    "tail": {"text": ent.text, "label": "ORG_SECUNDARIA"},
                    "relation": "CONTAINS",
                    "head_offsets": {"start": current_org.start_char, "end": current_org.end_char},
                    "tail_offsets": {"start": ent.start_char, "end": ent.end_char},
                })
            current_sub = ent

        elif ent.label_ == "DOC":
            if current_sub and is_company_doc_label_text(ent.text):
                doc._.relations.append({
                    "head": {"text": current_sub.text, "label": "ORG_SECUNDARIA"},
                    "tail": {"text": ent.text, "label": "DOC"},
                    "relation": "HAS_DOCUMENT",
                    "head_offsets": {"start": current_sub.start_char, "end": current_sub.end_char},
                    "tail_offsets": {"start": ent.start_char, "end": ent.end_char},
                })
            elif current_org:
                doc._.relations.append({
                    "head": {"text": current_org.text, "label": "ORG"},
                    "tail": {"text": ent.text, "label": "DOC"},
                    "relation": "SECTION_ITEM",
                    "head_offsets": {"start": current_org.start_char, "end": current_org.end_char},
                    "tail_offsets": {"start": ent.start_char, "end": ent.end_char},
                })


# ---------------- Pretty print ----------------
def extract_doc_block(doc_obj: Doc, doc_ent_end: int) -> str:
    next_starts = [
        e.start_char for e in doc_obj.ents
        if e.start_char > doc_ent_end and e.label_ in ("ORG", "ORG_SECUNDARIA", "DOC")
    ]
    end = min(next_starts) if next_starts else len(doc_obj.text)
    return doc_obj.text[doc_ent_end:end].strip()

def print_output(doc: Doc) -> None:
    print("Entities:")
    for e in sorted(doc.ents, key=lambda x: (x.start_char, -x.end_char)):
        print(f"{e.text}  ->  {e.label_}")

    print("\nRelations:")
    print(f"{'HEAD':<70} {'TAIL':<60} {'RELATION'}")
    print("-" * 140)
    for r in doc._.relations:
        print(f"{r['head']['text']:<70} {r['tail']['text']:<60} {r['relation']}")

    print("\nDOC blocks:")
    for r in doc._.relations:
        if r["tail"]["label"] == "DOC":
            tail_end = r["tail_offsets"]["end"]
            snippet = extract_doc_block(doc, tail_end)
            print("\nHEAD:", r["head"]["text"])
            print("DOC :", r["tail"]["text"])
            print("TEXT:")
            print(snippet)


# ---------------- Runner ----------------
def process_text(raw_text: str) -> None:
    text = normalize_text(raw_text)
    # Tokenizer only; models disabled to keep control
    nlp = spacy.load("pt_core_news_lg", disable=["ner", "tagger", "parser", "lemmatizer"])
    doc = nlp.make_doc(text)

    # Detect entities with structural rules
    doc.ents = detect_entities(doc)

    # Build relations
    build_relations(doc)

    # Print output
    print_output(doc)



content="""
VICE-PRESIDÊNCIA DO GOVERNO REGIONAL
Rectificação
SECRETARIAREGIONAL DOS RECURSOS HUMANOS
Avisos
SECRETARIAREGIONAL DO TURISMO E CULTURA
Avisos
SECRETARIAREGIONAL DO EQUIPAMENTO SOCIAL E TRANSPORTES,
Aviso
CONSERVATÓRIA DO REGISTO COMERCIAL DO FUNCHAL
ACTION LASER - INFORMÁTICA, LIMITADA
Contrato de sociedade
JETMADEIRA - EQUIPAMENTO NÁUTICO, LIMITADA
Contrato de sociedade
MADIGAB - GABINETE DE ENGENHARIAE FISCALIZAÇÃO DE OBRAS DA
MADEIRA, LIMITADA
Contrato de sociedade
RALNEC - VESTUÁRIO, LIMITADA
Contrato de sociedade

"""

process_text(content)

