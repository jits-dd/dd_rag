from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.postprocessor import LLMRerank
from typing import List
from config.settings import settings
import logging
from pymilvus import (
    connections, utility, Collection,
    FieldSchema, CollectionSchema, DataType
)

class AdvancedConversationRetriever(BaseRetriever):
    def __init__(self, vector_store, embed_model):
        super().__init__()
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.logger = logging.getLogger(__name__)

        # Ensure collection is loaded
        connections.connect(
            alias="default",
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            user=settings.MILVUS_USER,
            password=settings.MILVUS_PASSWORD
        )
        self.collection = Collection(vector_store.collection_name)
        self.collection.load()

        self.reranker = LLMRerank(
            llm=settings.llm,
            top_n=settings.RERANK_TOP_K
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Complete retrieval implementation with direct Milvus access"""
        try:
            query_str = str(query_bundle.query_str)
            print(f"\nStarting retrieval for: {query_str}")

            # 1. Generate embedding
            query_embedding = self.embed_model.get_text_embedding(query_str)
            print(f"Generated embedding (dim: {len(query_embedding)})")

            # 2. Direct Milvus query
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }

            print("Executing Milvus search...")
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",  # Use direct field name
                param=search_params,
                limit=settings.RETRIEVAL_TOP_K * 2,
                output_fields=["text", "metadata", "file_name", "title", "summary"]
            )

            print(f"Found {len(results[0])} raw results")

            # 3. Convert to nodes
            nodes = []
            for hit in results[0]:
                print(f"Hit printing - {hit}")
                nodes.append(NodeWithScore(
                    node=TextNode(
                        text=hit.entity.get("text"),
                        metadata=hit.entity.get("metadata"),
                        embedding=hit.entity.get("embedding")
                    ),
                    score=hit.score
                ))

            # 4. Rerank if we have results
            if nodes:
                print(f"Reranking {len(nodes)} nodes...")
                nodes = self.reranker.postprocess_nodes(
                    nodes,
                    query_bundle=query_bundle
                )

            return nodes[:settings.RETRIEVAL_TOP_K]

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