import os
import json
import numpy as np
import faiss
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
token_encoder = tiktoken.encoding_for_model("text-embedding-ada-002")

class RagSearcher:
    def __init__(self, json_path="chunks.json"):
        self.index = None
        self.chunks = []
        self.embeddings = []
        self._build_index(json_path)

    def _embed(self, text):
        if not text or not text.strip():
            print("⚠️ 空のテキストをスキップしました")
            return np.zeros(1536, dtype=np.float32)
        # トークン数制限（長すぎるチャンクはカット）
        tokens = token_encoder.encode(text)
        if len(tokens) > 8192:
            text = token_encoder.decode(tokens[:8192])
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _build_index(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        texts = [chunk["text"] for chunk in self.chunks]
        self.embeddings = [self._embed(text) for text in texts]
        dim = len(self.embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))

    def search(self, query, top_k=3):
        query_vec = self._embed(query).reshape(1, -1)
        distances, indices = self.index.search(query_vec, top_k)
        return [self.chunks[i]["text"] for i in indices[0]]

    def search_filtered(self, query, role=None, location=None, top_k=3):
        filtered_chunks = []

        for chunk in self.chunks:
            if role and "role" in chunk and role not in chunk["role"]:
                continue
            if location and "location" in chunk and location not in chunk["location"]:
                continue
            filtered_chunks.append(chunk)

        if not filtered_chunks:
            print("⚠️ 該当チャンクがありません")
            return ["（該当する情報がマニュアル内に見つかりませんでした）"]

        texts = [chunk["text"] for chunk in filtered_chunks]
        embeddings = [self._embed(text) for text in texts]
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        query_vec = self._embed(query).reshape(1, -1)
        distances, indices = index.search(query_vec, min(top_k, len(embeddings)))
        return [texts[i] for i in indices[0]]
