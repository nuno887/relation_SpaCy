# main.py
import spacy
from spacy.tokens import Doc
from spacy.language import Language

from spacy.matcher import RegexMatcher
from spacy.tokens import Span

# Load Portuguese pipeline but disable built-in NER (we'll control entities via EntityRuler)
nlp = spacy.load("pt_core_news_lg", disable=["ner"])

# --- EntityRuler with TOKEN PATTERNS (robust to newlines/whitespace) ---
ruler = nlp.add_pipe(
    "entity_ruler",
    config={"overwrite_ents": True}  # ruler outputs become the entities
)

patterns = [
    # ORGs as TOKEN PATTERNS (case-insensitive via LOWER)
    {
        "label": "ORG",
        "pattern": [
            {"LOWER": "secretaria"},
            {"LOWER": "regional"},
            {"LOWER": "das"},
            {"LOWER": "finanças"},
        ],
    },
    # Case A: spaCy tokenizes "VICE-PRESIDÊNCIA" as a single token
    {
        "label": "ORG",
        "pattern": [
            {"LOWER": "vice-presidência"},
            {"LOWER": "do"},
            {"LOWER": "governo"},
            {"LOWER": "regional"},
            {"LOWER": "e"},
            {"LOWER": "dos"},
            {"LOWER": "assuntos"},
            {"LOWER": "parlamentares"},
        ],
    },
    # Case B: spaCy splits into "VICE", optional "-", "PRESIDÊNCIA"
    {
        "label": "ORG",
        "pattern": [
            {"LOWER": "vice"},
            {"ORTH": "-", "OP": "?"},
            {"LOWER": "presidência"},
            {"LOWER": "do"},
            {"LOWER": "governo"},
            {"LOWER": "regional"},
            {"LOWER": "e"},
            {"LOWER": "dos"},
            {"LOWER": "assuntos"},
            {"LOWER": "parlamentares"},
        ],
    },

    # Generic DOC pattern: "Revogação n.º 74/2022" (supports n.º / nº / n° / n.o)
    {
        "label": "DOC",
        "pattern": [
            {"LOWER": "revogação"},
            {"LOWER": {"IN": ["n.º", "nº", "n°", "n.o"]}},
            {"TEXT": {"REGEX": r"^\d+/\d{4}$"}},
        ],
    },
]
ruler.add_patterns(patterns)

# --- Relation storage ---
Doc.set_extension("relations", default=[], force=True)

# --- Relation component: link each DOC to the most recent ORG seen before it ---
@Language.component("rel_org_doc")
def rel_org_doc(doc: Doc):
    doc._.relations = []
    current_org = None

    # iterate entities in document order; token patterns ensure multi-line ORGs match
    for ent in doc.ents:
        if ent.label_ == "ORG":
            current_org = ent
        elif ent.label_ == "DOC" and current_org:
            doc._.relations.append({
                "head": {"text": current_org.text, "label": current_org.label_},
                "tail": {"text": ent.text, "label": ent.label_},
                "relation": "ISSUES",
                # Offsets so we can extract the DOC block text later
                "head_offsets": {"start": current_org.start_char, "end": current_org.end_char},
                "tail_offsets": {"start": ent.start_char, "end": ent.end_char},
            })
    return doc

nlp.add_pipe("rel_org_doc", last=True)

# --- Demo text (includes a multi-line ORG name) ---
text = """
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
"""

doc = nlp(text)

# Helper: extract the text under a DOC heading up to the next ORG/DOC or end of doc
def extract_doc_block(doc_obj, doc_ent_end):
    """Return the text from the end of a DOC entity until the next ORG/DOC or end of document."""
    next_starts = [
        e.start_char for e in doc_obj.ents
        if e.start_char > doc_ent_end and e.label_ in ("ORG", "DOC")
    ]
    end = min(next_starts) if next_starts else len(doc_obj.text)
    return doc_obj.text[doc_ent_end:end].strip()

# --- Print entities (for debugging) ---
print("Entities:")
for ent in doc.ents:
    print(f"{ent.text}  ->  {ent.label_}")

# --- Pretty-print relations in a table ---
print("\nRelations:")
print(f"{'ORG':<70} {'DOC':<25} {'RELATION'}")
print("-" * 110)
for r in doc._.relations:
    org = r['head']['text']
    doc_id = r['tail']['text']
    rel = r['relation']
    print(f"{org:<70} {doc_id:<25} {rel}")

# --- ORG / DOC / TEXT blocks ---
print("\nDOC blocks:")
for r in doc._.relations:
    block_text = extract_doc_block(doc, r["tail_offsets"]["end"])
    print("\nORG:", r["head"]["text"])
    print("DOC:", r["tail"]["text"])
    print("TEXT:")
    print(block_text)
