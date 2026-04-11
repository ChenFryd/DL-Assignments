#!/usr/bin/env python
import sys


def extract_text_with_pdfplumber(pdf_path: str) -> str:
    import pdfplumber

    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


def extract_text_with_pypdf2(pdf_path: str) -> str:
    from PyPDF2 import PdfReader

    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


def extract_text_with_pypdf(pdf_path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: extract_pdf.py <pdf_path> [output_txt_path]", file=sys.stderr)
        return 1

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    extractors = [
        ("pdfplumber", extract_text_with_pdfplumber),
        ("PyPDF2", extract_text_with_pypdf2),
        ("pypdf", extract_text_with_pypdf),
    ]

    last_error = None
    full_text = ""

    for name, extractor in extractors:
        try:
            full_text = extractor(pdf_path)
            break
        except ImportError:
            print(f"{name} not found, trying next library...", file=sys.stderr)
        except Exception as exc:
            last_error = f"Error with {name}: {exc}"
            print(last_error, file=sys.stderr)

    if not full_text and last_error:
        print("No suitable PDF library succeeded", file=sys.stderr)
        return 1

    if output_path:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(full_text)
    else:
        print(full_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())