import sys
import unicodedata
import spacy
from spacy.tokens import Doc, Span
from spacy.language import Language
from spacy.matcher import Matcher


# ---------------- Load Portuguese pipeline (disable built-in NER) ----------------
nlp = spacy.load("pt_core_news_lg", disable=["ner"])

ALLOWED_ORG_START = {"SECRETARIA", "VICE", "PRESIDÊNCIA", "PRESIDENCIA"}
DISALLOWED_ORG_START = {"REVOGAÇÃO"}  # never treat these as ORG headers

# ---------------- EntityRuler for DOC only ----------------
ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
ruler.add_patterns([
    {
        "label": "DOC",
        "pattern": [
            {"LOWER": "revogação"},
            {"LOWER": {"IN": ["n.º", "nº", "n°", "n.o"]}},
            {"TEXT": {"REGEX": r"^\d+/\d{4}$"}},
        ],
    }
])

# ---------------- ORG via Matcher anchored to line starts ----------------
matcher = Matcher(nlp.vocab)

# Start-of-line anchor:
# - Case 1: newline token followed by an uppercase token
# - Case 2: doc start (no preceding newline) with an uppercase token
matcher.add("ORG_LINE_START", [
    [{"ORTH": "\n"}, {"IS_UPPER": True}],
    [{"IS_UPPER": True}],  # doc-start case
])

@Language.component("org_from_matcher")
def org_from_matcher(doc: Doc):
    """Find ALL-CAPS headers that start at the beginning of a line and turn them into ORG spans.
       Only accept headers whose FIRST alpha token is in ALLOWED_ORG_START.
       Span can include consecutive ALL-CAPS lines."""
    matches = matcher(doc)
    spans = []

    def is_caps_token(tok):
        if tok.is_space:
            return True
        if tok.is_punct and tok.text in "-&/.,()«»“”\"'":
            return True
        letters = "".join(ch for ch in tok.text if ch.isalpha())
        return bool(letters) and letters.isupper()

    for _, start, end in matches:
        # normalize to first non-newline token
        i = start
        if doc[start].text == "\n" and start + 1 < len(doc):
            i = start + 1

        # guard: first alpha token must be allowed (and not disallowed)
        first_alpha = None
        k = i
        while k < len(doc) and doc[k].text == "\n":
            k += 1
        while k < len(doc) and (doc[k].is_space or not doc[k].is_alpha):
            k += 1
        if k < len(doc):
            first_alpha = doc[k].text.upper()

        if (first_alpha is None
            or first_alpha in DISALLOWED_ORG_START
            or first_alpha not in ALLOWED_ORG_START):
            continue  # skip this header

        # extend over consecutive ALL-CAPS "header lines"
        j = i
        seen_alpha = 0
        while j < len(doc):
            # stop if this line is not header-ish
            if doc[j].text == "\n":
                # peek next line's first non-space token to decide if we continue
                t = j + 1
                # skip spaces/newlines
                while t < len(doc) and (doc[t].is_space and doc[t].text != "\n"):
                    t += 1
                # skip the newline itself
                if t < len(doc) and doc[t].text == "\n":
                    j = t
                    continue
                # find first token of next line
                while t < len(doc) and doc[t].is_space:
                    t += 1
                # if next line starts with caps token, continue; else break
                if t < len(doc) and is_caps_token(doc[t]) and doc[t].is_alpha:
                    j += 1  # include the newline and continue scanning
                    continue
                else:
                    break

            if not is_caps_token(doc[j]):
                break
            if doc[j].is_alpha:
                seen_alpha += 1
            j += 1

        if seen_alpha >= 2 and j > i:
            span = Span(doc, i, j, label=doc.vocab.strings["ORG"])
            if span.text.strip():
                spans.append(span)

    doc.ents = spacy.util.filter_spans(list(doc.ents) + spans)
    return doc

# Add ORG detector after the EntityRuler so DOCs are already present
nlp.add_pipe("org_from_matcher", after="entity_ruler")

# ---------------- Relations: each DOC -> most recent preceding ORG ----------------
Doc.set_extension("relations", default=[], force=True)

@Language.component("rel_org_doc")
def rel_org_doc(doc: Doc):
    doc._.relations = []
    current_org = None
    for ent in sorted(doc.ents, key=lambda e: e.start_char):
        if ent.label_ == "ORG":
            current_org = ent
        elif ent.label_ == "DOC" and current_org:
            doc._.relations.append({
                "head": {"text": current_org.text, "label": current_org.label_},
                "tail": {"text": ent.text, "label": ent.label_},
                "relation": "ISSUES",
                # offsets for block extraction
                "head_offsets": {"start": current_org.start_char, "end": current_org.end_char},
                "tail_offsets": {"start": ent.start_char, "end": ent.end_char},
            })
    return doc

nlp.add_pipe("rel_org_doc", last=True)

# ADD (after rel_org_doc component)
@Language.component("sumario")
def rel_org_org_same(doc: Doc):
    # group ORGs by normalized name
    by_key = {}
    for ent in doc.ents:
        if ent.label_ == "ORG":
            key = normalize_org_name(ent.text)
            if key:
                by_key.setdefault(key, []).append(ent)

    # for each name that appears more than once, link neighbors to avoid O(n^2) blowup
    for key, spans in by_key.items():
        if len(spans) < 2:
            continue
        spans = sorted(spans, key=lambda e: e.start_char)
        for a, b in zip(spans, spans[1:]):
            doc._.relations.append({
                "head": {"text": a.text, "label": "ORG"},
                "tail": {"text": b.text, "label": "ORG"},
                "relation": "SUMARIO",
                "head_offsets": {"start": a.start_char, "end": a.end_char},
                "tail_offsets": {"start": b.start_char, "end": b.end_char},
            })
    return doc

# ADD to pipeline (after existing relation builder)
nlp.add_pipe("sumario", last=True)


# ---------------- Helper: extract the text under a DOC heading ----------------
def extract_doc_block(doc_obj: Doc, doc_ent_end: int) -> str:
    """Return text from the end of a DOC entity until the next ORG/DOC or end of document."""
    next_starts = [
        e.start_char for e in sorted(doc_obj.ents, key=lambda e: e.start_char)
        if e.start_char > doc_ent_end and e.label_ in ("ORG", "DOC")
    ]
    end = min(next_starts) if next_starts else len(doc_obj.text)
    return doc_obj.text[doc_ent_end:end].strip()

def normalize_org_name(s: str) -> str:
    # accent-insensitive, case-insensitive, whitespace/punct compacted
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # strip accents
    s = s.upper()
    keep = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            keep.append(ch)
    s = "".join(keep)
    s = " ".join(s.split())  # collapse spaces
    return s

# ---------------- Optional: normalize hidden Unicode junk before tokenization ----------------
def normalize_text(s: str) -> str:
    # Canonicalize, remove BOM, soft hyphen, zero-width space, convert NBSP to regular space
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\ufeff", "").replace("\u00ad", "").replace("\u200b", "")
    s = s.replace("\u00a0", " ")
    return s

# ---------------- Public function to process any text ----------------
def process_text(raw_text: str) -> None:
    doc = nlp(normalize_text(raw_text))

    # Entities (debug)
    print("Entities:")
    for ent in sorted(doc.ents, key=lambda e: e.start_char):
        print(f"{ent.text}  ->  {ent.label_}")

    # Relations table
    print("\nRelations:")
    print(f"{'ORG':<70} {'DOC':<25} {'RELATION'}")
    print("-" * 110)
    for r in doc._.relations:
        print(f"{r['head']['text']:<70} {r['tail']['text']:<25} {r['relation']}")

    # ORG / DOC / TEXT blocks
    print("\nDOC blocks:")
    for r in doc._.relations:
        block_text = extract_doc_block(doc, r["tail_offsets"]["end"])
        print("\nORG:", r["head"]["text"])
        print("DOC:", r["tail"]["text"])
        print("TEXT:")
        print(block_text)

    # --- SUMARIO (ORG ↔ ORG with same name) ---
    print("\nSUMARIO relations (same ORG names):")
    print(f"{'ORG A':<60} {'ORG B':<60} {'RELATION'}")
    print("-" * 140)
    for r in doc._.relations:
        if r.get("relation") == "SUMARIO":
            a = r["head"]["text"]
            b = r["tail"]["text"]
            print(f"{a:<60} {b:<60} {r['relation']}")



content = """
SECRETARIA REGIONAL DAS 
FINANÇAS
Revogação n.º 338/2025
Revoga a autorização concedida por despacho do então Vice-Presidente do Governo
Regional datado de 28/05/2019, da sociedade BEATLANTIC, LDA., pela sua
dissolução e encerramento da liquidação no ano de 2021.
Revogação n.º 339/2025
Revoga a autorização concedida por despacho do então Secretário Regional do
Plano e Finanças datado de 29/12/2000, da sociedade FOOTBRIDGE -
- CONSULTORES E SERVIÇOS, LDA., pela sua dissolução e encerramento da
liquidação no ano de 2022.
Revogação n.º 340/2025
Revoga a autorização concedida por despacho do então Vice-Presidente do Governo
Regional em 23/08/2018, da sociedade GLOSEPRO - GLOBAL SERVICE
PROVIDER, UNIPESSOAL, LDA., pela sua dissolução e encerramento da
liquidação no ano de 2024.
Revogação n.º 341/2025
Revoga a autorização concedida por despacho do então Secretário Regional do
Plano e Finanças datado de 15/11/2011, da sociedade GZ ELECTRONICS, S.A.,
pela sua dissolução e encerramento da liquidação no ano de 2021.
Revogação n.º 342/2025
Revoga a autorização concedida por despacho do então Secretário Regional do
Plano e Finanças datado de 23/08/2018, da sociedade LA ROSSA -
- INTERNACIONAL, CONSULTORIA E SERVIÇOS, LDA., pela sua dissolução e
encerramento da liquidação no ano de 2020.
Revogação n.º 343/2025
Revoga a autorização concedida por despacho do então Secretário Regional do
Plano e Finanças datado de 24/11/2000, da sociedade MELK - COMÉRCIO E
SERVIÇOS INTERNACIONAIS, UNIPESSOAL, LDA., pela sua dissolução e
encerramento da liquidação no ano de 2022.
Revogação n.º 344/2025
Revoga a autorização concedida por despacho do então Secretário Regional das
Finanças e da Administração Pública datado de 21/10/2016, da sociedade
MESOPRO ESTHETICS, LDA., pela sua dissolução e encerramento da liquidação
no ano de 2025.
Revogação n.º 345/2025
Revoga a autorização concedida por despacho do então Secretário Regional do
Plano e Finanças datado de 19/12/2014, da sociedade PRIMEIRA POSIÇÃO,
SOCIEDADE UNIPESSOAL, LDA., pela sua dissolução e encerramento da
liquidação no ano de 2022.

VICE - PRESIDÊNCIA DO GOVERNO REGIONAL E DOS ASSUNTOS PARLAMENTARES
Revogação n.º 74/2022
Revoga a autorização concedida pelo então Secretário Regional do Plano e da
Coordenação em 04/05/1999, para o exercício da atividade da sociedade
“LOVIENT - CONSULTADORIA E SERVIÇOS, SOCIEDADE UNIPESSOAL, LDA.”.
Revogação n.º 75/2022
Revoga a autorização concedida pelo então Secretário Regional do Plano e da
Coordenação em 04/07/2000, para o exercício da atividade da sociedade “VAL-ENT
TRADING E INVESTIMENTOS, SOCIEDADE UNIPESSOAL, LDA.”.
Revogação n.º 76/2022
Revoga a autorização concedida pelo então Secretário Regional do Plano e da
Coordenação em 28/12/1999, para o exercício da atividade da sociedade
“TRIUMPHAL - SERVIÇOS DE CONSULTADORIA, LDA.”.
Revogação n.º 77/2022
Revoga a autorização concedida pelo então Secretário Regional do Plano e Finanças
em 15/11/2012, para o exercício da atividade da sociedade “PEDRA
ESMERIZ - CONSTRUÇÕES E SERVIÇOS, UNIPESSOAL, LDA.”.
Revogação n.º 78/2022
Revoga a autorização concedida pelo Vice-Presidente do Governo Regional em
13/02/2019, para o exercício da atividade da sociedade “EYAMBI - INTERNATIONAL
TRADING, LDA.”.
Revogação n.º 79/2022
Revoga a autorização concedida pelo então Secretário Regional das Finanças e da
Administração Pública em 06/08/2015, para o exercício da atividade da sociedade
“GLOBALCIVIL, S.A.”.
Revogação n.º 80/2022
Revoga a autorização concedida pelo então Secretário Regional do Plano e da
Coordenação em 06/01/1999, para o exercício da atividade da sociedade
“PRESEUS - SGPS, LDA”.
Revogação n.º 81/2022
Revoga a autorização concedida pelo então Secretário Regional de Economia e
Cooperação Externa em 04/07/1996, para o exercício da atividade da sociedade
“TRENTON COMMERCIAL & FINANCIAL BUSINESS, S.A.”.
Revogação n.º 82/2022
Revoga a autorização concedida pelo então Secretário Regional das Finanças e da
Administração Pública em 08/02/2017, para o exercício da atividade da sociedade
“FLOWLOOP CONSULTING UNIPESSOAL, LDA”.


SECRETARIA REGIONAL DAS FINANÇAS
"""
  
process_text(content)
    
