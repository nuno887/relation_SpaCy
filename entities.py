
import sys
import re
import unicodedata
from typing import List, Tuple, Optional
import spacy
from spacy.tokens import Doc, Span

from relations import build_relations


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
    "CONVOCATÓRIA", "CONVOCATORIA", "REVOGAÇÃO", "REVOGACAO","CONTRATO", "DECRETO", "RESOLUÇÃO", "RESOLUCAO", "DECRETO REGULAMENTAR REGIONAL", "PORTARIA"
}

# Company-level doc anchor (used for look-ahead or simple detection)
RX_CONTRATO_SOC = re.compile(r"(?is)\bcontrato\s*de\s*sociedade\b")

# Optional “institutional” starters for secondary orgs (used inside sections)
SECONDARY_STARTERS = {"INSTITUTO", "ASSOCIAÇÃO", "ASSOCIACAO", "CLUBE", "FUNDAÇÃO", "FUNDACAO", "DIREÇÃO", "DIRECÇÃO", "CLAQUE"}

# Content nouns that commonly appear on continuation lines of multi-line headers
CONTINUATION_CONTENT_NOUNS = {
    "PLANO", "FINANÇAS", "FINANCAS", "EDUCAÇÃO", "EDUCACAO", "RECURSOS", "HUMANOS",
    "CULTURA", "TURISMO", "TRANSPORTES", "AMBIENTE", "ASSUNTOS", "SOCIAIS"
}


# ---------------- Normalization & helpers ----------------

def has_lowercase_letter(line: str) -> bool:
    s = line.strip()
    return any(ch.isalpha() and ch.islower() for ch in s)

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
    # allow multiword starters at the very beginning (e.g., "PRESIDÊNCIA DO GOVERNO")
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
    if head.startswith(("DESPACHO", "DECLARAÇÃO", "DECLARACAO", "RETIFICAÇÃO", "RECTIFICAÇÃO", "AVISO", "AVISOS", "EDITAL", "ANÚNCIO", "ANUNCIO", "REVOGAÇÃO","REVOGACAO","CONTRATO","DECRETO", "RESOLUÇÃO", "RESOLUCAO", "PORTARIA")):
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
    """Continuation cues: current starts with stopword + has content nouns; or prev ends with connector/comma/hyphen."""
    if is_blank(curr_line):
        return False
    curr_up = strip(curr_line).upper()
    parts = curr_up.split()

    # starts with a function word and has content after
    if parts and parts[0] in STOPWORDS_UP and content_token_count(curr_up) >= 1:
        return True

    # previous ends with a joiner/comma/hyphen
    prev_up = strip(prev_line).upper()
    if prev_up.endswith((" E", " DO", " DA", " DE", " DOS", " DAS")):
        return True
    if prev_up.endswith((",", "-", "–")):
        return True

    # domain nouns after a stopword (e.g., "DO PLANO E FINANÇAS", "E DOS ASSUNTOS SOCIAIS")
    if parts and parts[0] in STOPWORDS_UP:
        if any(noun in curr_up for noun in CONTINUATION_CONTENT_NOUNS):
            return True

    return False

def looks_like_secondary_start(line: str) -> bool:
    """Secondary orgs often start with institutional nouns inside a section."""
    up = strip(line).upper()
    if not up:
        return False
    first = first_alpha_word_upper(up)
    return first in SECONDARY_STARTERS

def collapse_ws_for_display(text: str) -> str:
    """Collapse internal whitespace/newlines for pretty printing."""
    return " ".join(text.split())

def is_all_caps_line(line: str) -> bool:
    """True if the line (letters only) is ALL CAPS (Unicode-aware)."""
    s = strip(line)
    if not s:
        return False
    base = unicodedata.normalize("NFKD", s)
    base = "".join(ch for ch in base if not unicodedata.combining(ch))
    letters = [ch for ch in base if ch.isalpha()]
    if not letters:
        return False
    return all(ch == ch.upper() for ch in letters)


# ---------------- Span builder ----------------
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
    header_start = None
    header_end = None
    max_header_lines = 4  # covers 2–3 line headers safely

    def close_header():
        nonlocal header_start, header_end, spans
        if header_start is not None and header_end is not None and header_end > header_start:
            sp = char_span(doc, header_start, header_end, "ORG")
            if sp:
                spans.append(sp)
        header_start = header_end = None

    while i < len(lines):
        start, end, ln = lines[i]
        if state == "OUTSIDE":
            if is_blank(ln):
                i += 1
                continue
            if starts_with_header_starter(ln) and not has_lowercase_letter(ln):
                # start header and absorb continuations (up to cap)
                header_start, header_end = start, end
                header_lines = 1
                j = i + 1
                prev_ln = ln
                while j < len(lines) and header_lines < max_header_lines:
                    _, end_j, ln_j = lines[j]
                    if is_blank(ln_j) or is_doc_label_line(ln_j) or starts_with_header_starter(ln_j):
                        break
                    if is_header_continuation(prev_ln, ln_j):
                        header_end = end_j
                        prev_ln = ln_j
                        header_lines += 1
                        j += 1
                    else:
                        break
                i = j
                close_header()
                state = "IN_SECTION"
                continue
            else:
 #               if is_doc_label_line(ln):
  #                  sp = char_span(doc, start, end, "DOC")
   #                if sp:
   #                     spans.append(sp)
                i += 1
                continue

        # IN_SECTION
        if starts_with_header_starter(ln) and not is_blank(ln) and not has_lowercase_letter(ln):
            # new header
            header_start, header_end = start, end
            header_lines = 1
            j = i + 1
            prev_ln = ln
            while j < len(lines) and header_lines < max_header_lines:
                _, end_j, ln_j = lines[j]
                if is_blank(ln_j) or is_doc_label_line(ln_j) or starts_with_header_starter(ln_j) or has_lowercase_letter(ln_j):
                    break
                if is_header_continuation(prev_ln, ln_j):
                    header_end = end_j
                    prev_ln = ln_j
                    header_lines += 1
                    j += 1
                else:
                    break
            i = j
            close_header()
            state = "IN_SECTION"
            continue

        if is_doc_label_line(ln):
            sp = char_span(doc, start, end, "DOC")
            if sp:
                spans.append(sp)
            i += 1
            continue

        # Decide ORG_SECUNDARIA
        promote_secondary = False
        if not is_blank(ln) and not has_lowercase_letter(ln):
            if content_token_count(ln) >= 4 and not starts_with_header_starter(ln):
                promote_secondary = True
            else:
                # look-ahead for "Contrato de sociedade" within 2 lines
                j = i + 1
                steps = 0
                while steps < 2 and j < len(lines):
                    la_line = strip(lines[j][2]).upper()
                    if RX_CONTRATO_SOC.search(la_line):
                        promote_secondary = True
                        break
                    if starts_with_header_starter(la_line) or is_doc_label_line(la_line):
                        break
                    steps += 1
                    j += 1

        if promote_secondary:
            # --- NEW: eat 0–2 continuation lines first, then create ONE span ---
            block_start, block_end = start, end
            j = i + 1
            consumed = 0
            while consumed < 2 and j < len(lines):
                _, end_j, ln_j = lines[j]
                if is_blank(ln_j) or starts_with_header_starter(ln_j) or is_doc_label_line(ln_j) or has_lowercase_letter(ln_j):
                    break
                block_end = end_j
                j += 1
                consumed += 1

            sp = char_span(doc, block_start, block_end, "ORG_SECUNDARIA")
            if sp:
                spans.append(sp)
            i = j
            continue

        # plain text
        i += 1

    # --- NEW: dedupe exact spans, then resolve overlaps (keep longest) ---
    seen = set()
    uniq = []
    for sp in spans:
        key = (sp.start_char, sp.end_char, sp.label_)
        if key not in seen:
            seen.add(key)
            uniq.append(sp)
    return spacy.util.filter_spans(uniq)



# ---------------- Pretty print ----------------
def print_output(doc: Doc) -> None:
    print("Entities:")
    for e in sorted(doc.ents, key=lambda x: (x.start_char, -x.end_char)):
        label = e.label_
        txt = collapse_ws_for_display(e.text) if label == "ORG" else e.text
        print(f"{txt}\n  ->  {label}")
    
    print("\nRelations:")
    for rel in doc._.relations:
        head_label = rel["head"]["label"]
        head_text  = rel["head"]["text"]
        tail_label = rel["tail"]["label"]
        tail_text  = rel["tail"]["text"]
        relation   = rel["relation"]

        # keep ORG pretty (collapsed whitespace), like you did for entities
        if head_label == "ORG":
            head_text = collapse_ws_for_display(head_text)
        if tail_label == "ORG":
            tail_text = collapse_ws_for_display(tail_text)

        print(f"{head_label} -> {tail_label} {relation} | {head_text} -> {tail_text}")


"""
# ---------------- Runner ----------------
def process_text(raw_text: str) -> None:
    text = normalize_text(raw_text)
    # Tokenizer only; models disabled to keep control
    nlp = spacy.load("pt_core_news_lg", disable=["ner", "tagger", "parser", "lemmatizer"])
    doc = nlp.make_doc(text)

    # Detect entities with structural rules
    doc.ents = detect_entities(doc)

    build_relations(doc)

    # Print output (relations intentionally omitted for now)
    print_output(doc)

"""