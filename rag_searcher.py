# rag_searcher.py
import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RagSearcher:
    def __init__(self, json_path="chunks.json"):
        self.index = None
        self.chunks = []
        self.embeddings = []
        self._build_index(json_path)

    def _embed(self, text):
        if not text or not text.strip():
            print("⚠️ 空のテキストをスキップしました")
            return np.zeros(1536, dtype=np.float32)  # text-embedding-ada-002 のベクトルサイズ

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

