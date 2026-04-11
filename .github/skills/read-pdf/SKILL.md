---
name: read-pdf
description: 'Read and extract text from PDF files using Python. Use when asked to parse assignment instructions, summarize PDFs, or troubleshoot PDF text extraction issues.'
argument-hint: 'Provide PDF path and optional output file path'
user-invocable: true
---

# Read PDF

## When To Use
- A user asks to read or extract text from a PDF in the workspace.
- You need a fallback strategy across multiple Python PDF libraries.
- You need quick terminal-based extraction before doing analysis.

## Inputs
- PDF file path (required)
- Optional output file path for extracted text

## Procedure
1. Verify the PDF path exists.
2. Run [extract_pdf.py](./scripts/extract_pdf.py) with the PDF path.
3. If extraction succeeds, use the text directly or save it to a file.
4. If extraction fails due to missing libraries, install one dependency: `pypdf`.

## Command Examples
```powershell
# Print extracted text to terminal
py .github/skills/read-pdf/scripts/extract_pdf.py HW1_instructions.pdf

# Save extracted text to a file
py .github/skills/read-pdf/scripts/extract_pdf.py HW1_instructions.pdf HW1_instructions.txt
```

## Notes
- The script tries libraries in order: `pdfplumber`, `PyPDF2`, then `pypdf`.
- If no supported library is installed, install one with:
```powershell
py -m pip install pypdf
```