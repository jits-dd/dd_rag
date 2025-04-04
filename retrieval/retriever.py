from typing import List
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.postprocessor import LLMRerank
from config.settings import settings
import logging

class AdvancedConversationRetriever(BaseRetriever):
    def __init__(self, vector_store, embed_model):
        super().__init__()
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.logger = logging.getLogger(__name__)
        self.reranker = LLMRerank(
            llm=settings.llm,
            top_n=settings.RERANK_TOP_K
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        try:
            # Get initial dense retrieval results
            query_str = str(query_bundle.query_str)
            query_embedding = self.embed_model.get_text_embedding(query_str)

            # Search with higher recall initially
            dense_results = self.vector_store.query(
                query_embedding=query_embedding,
                similarity_top_k=settings.RETRIEVAL_TOP_K * 2,
                filters=None
            )

            if not dense_results:
                return []

            # Rerank with LLM for better precision
            reranked_nodes = self.reranker.postprocess_nodes(
                dense_results,
                query_bundle=query_bundle
            )

            # Prioritize conversation nodes when present
            final_nodes = self._prioritize_conversation_nodes(reranked_nodes)

            return final_nodes[:settings.RETRIEVAL_TOP_K]

        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return []

    def _prioritize_conversation_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Reorder results to prioritize conversation nodes when relevant"""
        conversation_nodes = [
            n for n in nodes
            if n.metadata.get("is_conversation", False)
        ]

        # If we found conversation nodes and they're reasonably relevant (score > 0.5)
        if conversation_nodes and conversation_nodes[0].score > 0.5:
            return conversation_nodes + [
                n for n in nodes
                if not n.metadata.get("is_conversation", False)
            ]
        return nodes