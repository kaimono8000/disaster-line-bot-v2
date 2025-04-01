import numpy as np
import faiss
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 入力ファイル（テキストから作ったやつ）
with open("LINEbot_chunks_actioncard_from_txt.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# ⚠️ チャンクが長すぎる時に分割する関数（3000文字で切る）
def split_text(text, max_chars=3000):
    chunks = []
    while len(text) > max_chars:
        split_at = text.rfind("。", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        chunks.append(text[:split_at + 1].strip())
        text = text[split_at + 1:]
    chunks.append(text.strip())
    return chunks

# OpenAI埋め込みAPI
def embed(text):
    if not text.strip():
        return np.zeros(1536, dtype=np.float32)
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# 最終チャンクとベクトル格納
final_chunks = []
embeddings = []

for chunk in chunks:
    parts = split_text(chunk["text"])
    for part in parts:
        vec = embed(part)
        embeddings.append(vec)
        final_chunks.append({
            **chunk,
            "text": part
        })

# FAISSインデックス作成
dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# 保存
faiss.write_index(index, "actioncard_index.faiss")
with open("actioncard_chunks_metadata.json", "w", encoding="utf-8") as f:
    json.dump(final_chunks, f, ensure_ascii=False, indent=2)

print("✅ FAISSとメタデータの保存完了！")
