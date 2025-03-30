import fitz  # PyMuPDF

def split_pdf_to_chunks(pdf_path, chunk_size=500, overlap=50):
    doc = fitz.open(pdf_path)
    chunks = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append({
                "text": chunk,
                "page": page_num + 1
            })
            start += chunk_size - overlap

    return chunks

