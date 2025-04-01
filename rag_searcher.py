import os
import json
import numpy as np
import faiss
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RagSearcher:
    def __init__(self, json_path="chunks.json", max_tokens=3000):
        self.index = None
        self.chunks = []
        self.chunk_metadatas = []
        self.embeddings = []
        self.tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
        self.max_tokens = max_tokens
        self._build_index(json_path)

    def _embed(self, text):
        if not text or not text.strip():
            print("⚠️ 空のテキストをスキップしました")
            return np.zeros(1536, dtype=np.float32)

        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _build_index(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)

        self.chunks = [chunk["text"] for chunk in raw_chunks]
        self.chunk_metadatas = raw_chunks
        self.embeddings = [self._embed(text) for text in self.chunks]

        dim = len(self.embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))

    def search_filtered(self, query, role=None, location=None, category=None, top_k=5):
        query_vec = self._embed(query).reshape(1, -1)
        distances, indices = self.index.search(query_vec, top_k * 5)  # ゆとりをもって検索

        filtered = []
        total_tokens = 0

        for i in indices[0]:
            chunk = self.chunk_metadatas[i]

            # フィルタ：ゆるめ（小文字化＋部分一致）
            if role and chunk.get("role") and role.lower() not in chunk["role"].lower():
                continue
            if location and chunk.get("location") and location.lower() not in chunk["location"].lower():
                continue
            if category and chunk.get("category") and category.lower() not in chunk["category"].lower():
                continue

            # トークン数で制限（context最大8192なので余裕持って3000くらい）
            token_count = len(self.tokenizer.encode(chunk["text"]))
            if total_tokens + token_count > self.max_tokens:
                break

            total_tokens += token_count
            filtered.append(chunk["text"])

            print("🔍 ヒット:", chunk.get("role"), chunk.get("location"), chunk["text"][:40])

        return filtered
