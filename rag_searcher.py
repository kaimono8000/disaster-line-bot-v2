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
            return np.zeros(1536, dtype=np.float32)

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

    def search_with_routing(self, query, top_k=3):
        # 「配置」関連の質問かどうか判定
        keywords = ["配置", "配属", "どこに行く", "役割", "行動", "行動場所", "担当", "部署"]
        if any(word in query for word in keywords):
            print("🔍 配置関連ワードが検出されました。人員配置表に絞って検索します。")

            # 人員配置っぽいチャンクを抜き出す
            filtered_chunks = [
                chunk for chunk in self.chunks
                if any(kw in chunk["text"] for kw in ["人員配置", "役割分担", "行動分担", "配置表", "所属"])
            ]

            if not filtered_chunks:
                print("⚠️ 配置チャンクが見つからなかったので通常検索します。")
                return self.search(query, top_k=top_k)

            texts = [chunk["text"] for chunk in filtered_chunks]
            embeds = [self._embed(t) for t in texts]
            temp_index = faiss.IndexFlatL2(len(embeds[0]))
            temp_index.add(np.array(embeds))

            query_vec = self._embed(query).reshape(1, -1)
            D, I = temp_index.search(query_vec, top_k)

            return [texts[i] for i in I[0]]

        # 通常検索
        return self.search(query, top_k=top_k)
