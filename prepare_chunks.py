import json
from pdf_chunker import split_pdf_to_chunks

pdf_path = "manual.pdf"  # ← PDFのファイル名が違ったらここ直してね
output_path = "chunks.json"

# チャンク化して保存
chunks = split_pdf_to_chunks(pdf_path)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"{len(chunks)} チャンクを {output_path} に保存したよ！")

