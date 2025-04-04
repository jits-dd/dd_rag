from typing import List, Dict, Any
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import hashlib
import time
import logging
from config import settings

class ConversationProcessor:
    def __init__(self, embed_model: BaseEmbedding):
        self.primary_embed_model = embed_model
        self.fallback_embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        self.logger = logging.getLogger(__name__)

    def _enhance_metadata(self, node: BaseNode) -> Dict[str, Any]:
        """Add conversation-specific metadata"""
        metadata = node.metadata.copy() if hasattr(node, "metadata") else {}

        # Extract conversation features
        text = node.get_content()
        metadata.update({
            "processing_version": "3.0",
            "chunk_hash": self._generate_content_hash(node),
            "source_type": metadata.get("source", "unknown"),
            "is_conversation": "conversation" in metadata.get("source", "").lower(),
            "num_speakers": len(metadata.get("speakers", set())),
            "word_count": len(text.split()),
            "timestamp": int(time.time())
        })

        return metadata

    # In processor.py - update the process_nodes method
    def process_nodes(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """Process nodes with fallback embedding"""
        processed_nodes = []

        try:
            # Process metadata and IDs
            for node in nodes:
                if not isinstance(node, TextNode):
                    node = TextNode.from_base_node(node)

                # Store doc_id in metadata
                metadata = self._enhance_metadata(node)
                metadata["doc_id"] = node.metadata.get("file_name", "")
                node.metadata = metadata
                node.id_ = self._generate_node_id(node)
                processed_nodes.append(node)

            # Get embeddings in batch
            texts = [node.get_content() for node in processed_nodes]
            try:
                embeddings = self.primary_embed_model.get_text_embedding_batch(texts)
                self.logger.info("Used primary embedding model")
            except Exception as e:
                self.logger.warning(f"Primary embedding failed, using fallback: {e}")
                embeddings = self.fallback_embed_model.get_text_embedding_batch(texts)

            # Ensure embeddings are properly formatted
            for node, embedding in zip(processed_nodes, embeddings):
                if not isinstance(embedding, list):
                    embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                node.embedding = embedding

            self.logger.info(f"Processed {len(processed_nodes)} nodes")
            return processed_nodes

        except Exception as e:
            self.logger.error(f"Error processing nodes: {e}")
            raise

    def _generate_node_id(self, node: BaseNode) -> str:
        """Generate consistent node ID"""
        content = node.get_content()
        metadata = str(node.metadata)
        return hashlib.sha256(f"{content}{metadata}".encode()).hexdigest()

    def _generate_content_hash(self, node: BaseNode) -> str:
        """Generate content hash for change detection"""
        return hashlib.md5(node.get_content().encode()).hexdigest()