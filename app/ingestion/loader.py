from pathlib import Path

DATA_DIR = Path("data")

def load_documents():
    docs = []
    for file in DATA_DIR.glob("*"):
        text = file.read_text()
        docs.append({"text": text, "source": file.name})
    return docs