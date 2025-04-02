from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.schema import BaseNode
from typing import List
from config.settings import settings

class AdvancedRetriever:
    def __init__(self, vector_store: MilvusVectorStore):
        self.index = VectorStoreIndex.from_vector_store(vector_store)

    def get_retriever(self, similarity_top_k: int = 5):
        """Create hybrid retriever with bm25 and vector search"""
        return VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k,
            vector_store_query_mode="hybrid",
            alpha=0.5  # balance between bm25 and vector search
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[BaseNode]:
        retriever = self.get_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query)