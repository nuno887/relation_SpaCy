# api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from tempfile import NamedTemporaryFile

# 1) OCR → text (virtual)
from pdf_ocr_extractor.core import load_settings, process_pdf_file, _apply_tesseract_env

# 2) Text → bundle (use your updated main.run_pipeline)
from extractors.extracting_01.main import run_pipeline

app = FastAPI(title="Gazette Extractor API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    tmp_path: Path | None = None
    try:
        # Save upload to a temp file (PyMuPDF needs a path)
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        # 1) OCR to virtual text (no disk writes)
        cfg = load_settings()
        _apply_tesseract_env(cfg)
        vf = process_pdf_file(tmp_path, cfg)  # -> VirtualTextFile

        if not vf.text.strip():
            raise HTTPException(status_code=422, detail="No text extracted from PDF")

        # 2) Run extracting_01 pipeline (returns body_doc, bundle)
        bundle = run_pipeline(vf.text, show_debug=False)

        print(bundle)
        print("passed here--------------------------------------------------------------------------------------------------------")

        return bundle

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


# uvicorn api:app --reload --host 0.0.0.0 --port 8080
# 