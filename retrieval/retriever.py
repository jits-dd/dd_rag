from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.postprocessor import LLMRerank
from typing import List
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
        """Enhanced retrieval with proper query handling"""
        try:
            print(f"\nStarting retrieval for query: {query_bundle.query_str}")
            query_embedding = self.embed_model.get_text_embedding(str(query_bundle.query_str))
            print("Generated query embedding")

            # Step 2: Query vector store
            print("Querying Milvus vector store...")
            results = self.vector_store.query(
                query_embedding=query_embedding,
                similarity_top_k=settings.RETRIEVAL_TOP_K * 2
            )
            print(f"Milvus returned {len(results)} results")

            if not results:
                return []

            # Step 3: Convert to nodes
            nodes = [
                NodeWithScore(
                    node=TextNode(
                        text=result.text,
                        metadata=result.metadata,
                        embedding=result.embedding
                    ),
                    score=result.score
                ) for result in results
            ]

            # Step 4: Rerank
            print("Reranking results...")
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes,
                query_bundle=query_bundle
            )

            return reranked_nodes[:settings.RETRIEVAL_TOP_K]

        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return []

    def _apply_business_rules(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Apply business-specific prioritization rules"""
        # Prioritize conversation nodes when present
        conv_nodes = [n for n in nodes if n.metadata.get("is_conversation", False)]

        if conv_nodes and conv_nodes[0].score > 0.5:
            return conv_nodes + [
                n for n in nodes
                if not n.metadata.get("is_conversation", False)
            ]

        return nodes[:settings.RETRIEVAL_TOP_K]