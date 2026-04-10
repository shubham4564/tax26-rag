"""
Extract text from explanatory documents.

Strategy:
1. Direct text extraction for digital PDFs.
2. OCR fallback for scanned PDFs when extracted text is sparse.

Requirements for OCR fallback:
- pytesseract
- Pillow
- PyMuPDF (fitz)
- Tesseract installed on system path
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover
    fitz = None

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    source_file: str
    output_text_file: str
    method: str
    chars_extracted: int


class DocumentTextExtractor:
    """Extract text from PDF documents with OCR fallback."""

    def __init__(self, min_direct_chars: int = 500):
        self.min_direct_chars = min_direct_chars
        self.output_dir = Path("data/processed/explanatory_text")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._logged_missing_direct_backend = False

    def run_directory(self, input_dir: Path) -> Dict:
        """Process all PDFs in a directory recursively."""
        pdf_files = sorted(input_dir.rglob("*.pdf"))
        results: List[ExtractionResult] = []

        for pdf_file in pdf_files:
            result = self.extract_pdf(pdf_file)
            results.append(result)

        summary = {
            "input_dir": str(input_dir),
            "pdf_count": len(pdf_files),
            "processed_count": len(results),
            "results": [r.__dict__ for r in results],
            "generated_at": datetime.now().isoformat(),
        }

        summary_path = self.output_dir / "extraction_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def extract_pdf(self, pdf_path: Path) -> ExtractionResult:
        """Extract text from a single PDF using direct extraction + OCR fallback."""
        direct_text = self._extract_direct_text(pdf_path)

        if len(direct_text.strip()) >= self.min_direct_chars:
            method = "direct"
            final_text = direct_text
        else:
            ocr_text = self._extract_ocr_text(pdf_path)
            if len(ocr_text.strip()) > len(direct_text.strip()):
                method = "ocr_fallback"
                final_text = ocr_text
            else:
                method = "direct_sparse"
                final_text = direct_text

        rel_path = pdf_path.as_posix().replace("/", "_").replace(":", "")
        out_path = self.output_dir / f"{rel_path}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        return ExtractionResult(
            source_file=str(pdf_path),
            output_text_file=str(out_path),
            method=method,
            chars_extracted=len(final_text),
        )

    def _extract_direct_text(self, pdf_path: Path) -> str:
        texts: List[str] = []
        direct_backend_found = False

        try:
            from pypdf import PdfReader  # type: ignore

            direct_backend_found = True
            reader = PdfReader(str(pdf_path))
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    texts.append(page_text)
            return "\n\n".join(texts)
        except Exception:
            # Fall through to pdfplumber when pypdf is unavailable or fails.
            pass

        if fitz:
            try:
                direct_backend_found = True
                doc = fitz.open(str(pdf_path))
                for page in doc:
                    page_text = page.get_text("text") or ""
                    if page_text:
                        texts.append(page_text)
                doc.close()
                if texts:
                    return "\n\n".join(texts)
            except Exception:
                pass

        try:
            import pdfplumber  # type: ignore

            direct_backend_found = True
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text:
                        texts.append(page_text)
        except Exception as exc:
            if direct_backend_found:
                logger.warning("Direct extraction failed for %s: %s", pdf_path, exc)
            elif not self._logged_missing_direct_backend:
                logger.warning(
                    "No direct PDF extraction backend available. Install pypdf or pdfplumber for better results."
                )
                self._logged_missing_direct_backend = True
            return ""

        return "\n\n".join(texts)

    def _extract_ocr_text(self, pdf_path: Path) -> str:
        if not fitz or not pytesseract or not Image:
            return ""

        texts: List[str] = []
        try:
            doc = fitz.open(str(pdf_path))
            for page in doc:
                # Render page to image for OCR.
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                mode = "RGBA" if pix.alpha else "RGB"
                image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(image)
                if page_text:
                    texts.append(page_text)
            doc.close()
        except Exception as exc:
            logger.warning("OCR extraction failed for %s: %s", pdf_path, exc)
            return ""

        return "\n\n".join(texts)


def main() -> None:
    extractor = DocumentTextExtractor(min_direct_chars=500)

    targets: List[Tuple[str, Path]] = [
        ("irs_publications", Path("data/raw/federal/irs_publications")),
        ("irs_form_instructions", Path("data/raw/federal/irs_form_instructions")),
    ]

    all_summaries: Dict[str, Dict] = {}
    for key, path in targets:
        if path.exists():
            all_summaries[key] = extractor.run_directory(path)

    summary_path = Path("data/processed/explanatory_text/summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
