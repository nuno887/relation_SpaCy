import sys
import spacy
from pathlib import Path

from entities import normalize_text, detect_entities, print_output
from relations import build_relations
import segmenter

print("Using segmenter from:", segmenter.__file__)
print(segmenter.build_sumario_and_body.__name__)

from body_refind import build_body_via_sumario_spacy


def run_pipeline(raw_text: str, show_debug: bool = False):
    """Return (doc, sumario, body_items) for a raw input string."""
    text = normalize_text(raw_text)

    # tokenizer only; keep control of entities ourselves
    nlp = spacy.load("pt_core_news_lg", disable=["ner", "tagger", "parser", "lemmatizer"])
    doc = nlp.make_doc(text)

    # 1) entities (rule-based, from your entities.py)
    doc.ents = detect_entities(doc)

    # 2) relations (from relations.py)
    build_relations(doc)

    # 3) segmentation (from segmenter.py)
    sumario, roster, body_text = segmenter.build_sumario_and_body(doc, include_local_details=False)
    print("Roster:", roster)
    print()
    print("body_text:", body_text)

    # I cutted the sumario in the "build_sumario_and_body" 
    doc = nlp.make_doc(body_text)

    print("--------------------------------")
    print("doc:", doc)
    print("--------------------------------")
    



    body_items = build_body_via_sumario_spacy(doc, roster, include_local_details=False)

    if show_debug:
        # optional: your existing entities/relations debug
        print_output(doc)

    return doc, sumario, body_items


def _preview_outputs(sumario, body_items, doc_len: int):
    print("=== SUMÁRIO ===")
    # show a short preview of the raw sumário text
    print(sumario or "(empty)")
    print("\nSumário ents:", {k: len(v) for k, v in sumario.ents.items()})
    print("Sumário relations:", len(sumario.relations))

    print("\n=== BODY (ORG → DOC slices) ===")
    if not body_items:
        print("(no body items)")
    for it in body_items:
        print(f"[{it.order_index}] {it.org_text} :: {it.doc_title}")
        print(f"slice [{it.slice_start}:{it.slice_end}] ({it.slice_end - it.slice_start} chars)")
        # show a tiny snippet
        #body_snip = it.slice_text.replace("\n", " ").strip()
        print(it or "(empty)")
        print("-" * 60)



 
input_path = Path("file_02.txt")
raw = input_path.read_text(encoding="utf-8")

doc, sumario, body_items = run_pipeline(raw_text=raw, show_debug=False)
_preview_outputs(sumario, body_items, len(doc.text))
