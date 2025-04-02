from typing import List
from llama_index.core.schema import BaseNode
from llama_index.core.embeddings import BaseEmbedding
import hashlib
import time

class DocumentProcessor:
    def __init__(self, embed_model: BaseEmbedding):
        self.embed_model = embed_model

    def process_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        current_time = int(time.time())
        """Add metadata, embeddings, and unique IDs"""
        # Generate consistent IDs
        for node in nodes:
            node.id_ = self._generate_node_id(node)

            # Ensure metadata exists
            if not hasattr(node, "metadata"):
                node.metadata = {}

            # Add processing metadata
            node.metadata.update({
                "processing_version": "2.0",
                "chunk_hash": self._generate_content_hash(node),
                "doc_type": node.metadata.get("doc_type", "unknown"),
                "source_file": node.metadata.get("file_name", "unknown")
            })

            # Add required Milvus fields
            setattr(node, "document_id", node.metadata.get("doc_id", ""))

        # Batch embed
        texts = [node.get_content() for node in nodes]
        embeddings = self.embed_model.get_text_embedding_batch(texts)
        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        return nodes

    def _generate_node_id(self, node: BaseNode) -> str:
        """Generate SHA256 ID from content and metadata"""
        content = node.get_content()
        metadata = str(node.metadata)
        return hashlib.sha256(f"{content}{metadata}".encode()).hexdigest()

    def _generate_content_hash(self, node: BaseNode) -> str:
        """Generate separate content hash for change detection"""
        return hashlib.md5(node.get_content().encode()).hexdigest()