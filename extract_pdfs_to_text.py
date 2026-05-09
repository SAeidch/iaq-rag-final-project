"""
extract_pdfs_to_text.py

Read PDF files from an input folder, extract text page by page,
save one .txt file per PDF, and also write a JSONL metadata file.

Why this script:
- Your existing RAG code already expects .txt files
- This gives you a clean first preprocessing step
- The JSONL file is useful later for metadata, citations, and debugging

Requirements:
    pip install pypdf

Usage:
    python extract_pdfs_to_text.py --input_dir ./papers --output_dir ./data/corpus
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


def clean_text(text: str) -> str:
    """
    Basic text cleaning for extracted PDF text.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive whitespace inside lines
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse too many blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def safe_filename(name: str) -> str:
    """
    Convert a filename stem into a safer text filename.

    Args:
        name: Original file stem.

    Returns:
        Sanitized filename stem.
    """
    name = re.sub(r"[^\w\-_. ]", "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name


def extract_pdf(pdf_path: Path) -> Dict:
    """
    Extract text from a PDF file page by page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dictionary containing extracted document content and metadata.
    """
    reader = PdfReader(str(pdf_path))

    pages: List[Dict] = []
    full_text_parts: List[str] = []

    for i, page in enumerate(reader.pages):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        page_text = clean_text(raw_text)

        pages.append(
            {
                "page_num": i + 1,
                "text": page_text,
                "char_count": len(page_text),
            }
        )

        if page_text:
            # Keep page markers in the text file for later traceability
            full_text_parts.append(f"[Page {i + 1}]\n{page_text}")

    full_text = "\n\n".join(full_text_parts).strip()

    return {
        "source_pdf": pdf_path.name,
        "title_guess": pdf_path.stem,
        "num_pages": len(reader.pages),
        "pages": pages,
        "full_text": full_text,
        "full_char_count": len(full_text),
    }


def write_txt(doc: Dict, output_dir: Path) -> Path:
    """
    Write the extracted full text to a .txt file.

    Args:
        doc: Extracted document dictionary.
        output_dir: Output directory for .txt files.

    Returns:
        Path to written .txt file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_name = safe_filename(doc["title_guess"]) + ".txt"
    txt_path = output_dir / txt_name

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(doc["full_text"])

    return txt_path


def append_jsonl_record(record: Dict, jsonl_path: Path) -> None:
    """
    Append one JSON record to a JSONL file.

    Args:
        record: Dictionary to write.
        jsonl_path: Path to the JSONL file.
    """
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PDF text to .txt and JSONL.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Folder containing PDF files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Folder where extracted .txt files and metadata will be saved.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "corpus_metadata.jsonl"

    # Start fresh each run
    if jsonl_path.exists():
        jsonl_path.unlink()

    print(f"Found {len(pdf_files)} PDF files.")
    print(f"Writing outputs to: {output_dir}")

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        doc = extract_pdf(pdf_path)
        txt_path = write_txt(doc, output_dir)

        metadata_record = {
            "source_pdf": doc["source_pdf"],
            "title_guess": doc["title_guess"],
            "num_pages": doc["num_pages"],
            "full_char_count": doc["full_char_count"],
            "txt_file": txt_path.name,
            "pages": [
                {
                    "page_num": p["page_num"],
                    "char_count": p["char_count"],
                }
                for p in doc["pages"]
            ],
        }

        append_jsonl_record(metadata_record, jsonl_path)

        print(
            f"  Pages: {doc['num_pages']}, chars: {doc['full_char_count']}, "
            f"saved: {txt_path.name}"
        )

    print("\nDone.")
    print(f"TXT files saved in: {output_dir}")
    print(f"Metadata JSONL saved in: {jsonl_path}")


if __name__ == "__main__":
    main()