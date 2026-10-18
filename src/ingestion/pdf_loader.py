import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any

class PDFIngestionNode:
    """
    Decoupled Node 1: Ingestion Node.
    Loads and sanitizes text from PDF documents while preserving page-level metadata.
    Supports PyMuPDF (fitz) with graceful fallback to pdfplumber/pypdfium2.
    """

    def __init__(self):
        pass

    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a PDF file and returns a list of page objects with text and metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        # Compute document SHA-256 for provenance tracking
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            doc_hash = hashlib.sha256(file_bytes).hexdigest()[:12]

        pages_data = []

        # Strategy 1: Try PyMuPDF (fitz)
        try:
            import fitz
            doc = fitz.open(file_path)
            total_pages = len(doc)
            for page_index in range(total_pages):
                page = doc[page_index]
                text = page.get_text("text").strip()
                if text:
                    pages_data.append({
                        "doc_id": doc_hash,
                        "source": path.name,
                        "page_number": page_index + 1,
                        "text": text,
                        "metadata": {
                            "total_pages": total_pages,
                            "char_count": len(text),
                            "engine": "pymupdf"
                        }
                    })
            doc.close()
            if pages_data:
                return pages_data
        except ImportError:
            pass
        except Exception:
            pass

        # Strategy 2: Fallback to pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                for page_index, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        pages_data.append({
                            "doc_id": doc_hash,
                            "source": path.name,
                            "page_number": page_index + 1,
                            "text": text.strip(),
                            "metadata": {
                                "total_pages": total_pages,
                                "char_count": len(text.strip()),
                                "engine": "pdfplumber"
                            }
                        })
            if pages_data:
                return pages_data
        except ImportError:
            pass
        except Exception:
            pass

        # Strategy 3: Fallback to pypdfium2
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(file_path)
            total_pages = len(pdf)
            for page_index in range(total_pages):
                page = pdf[page_index]
                textpage = page.get_textpage()
                text = textpage.get_text_range()
                if text and text.strip():
                    pages_data.append({
                        "doc_id": doc_hash,
                        "source": path.name,
                        "page_number": page_index + 1,
                        "text": text.strip(),
                        "metadata": {
                            "total_pages": total_pages,
                            "char_count": len(text.strip()),
                            "engine": "pypdfium2"
                        }
                    })
            if pages_data:
                return pages_data
        except Exception as e:
            raise RuntimeError(f"All PDF parsing engines failed: {str(e)}")

        return pages_data
