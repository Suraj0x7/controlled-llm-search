import os
from PyPDF2 import PdfReader

def load_document(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    elif file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

    else:
        raise ValueError("Only TXT and PDF supported")

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)
