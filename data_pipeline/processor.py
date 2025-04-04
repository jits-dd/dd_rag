from typing import List
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import hashlib
import logging
from config import settings

class ConversationProcessor:
    def __init__(self, embed_model: BaseEmbedding):
        self.embed_model = embed_model
        self.fallback_embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        self.logger = logging.getLogger(__name__)

    def process_nodes(self, nodes: List[BaseNode]) -> List[TextNode]:
        """Process nodes with fallback embedding"""
        processed_nodes = []
        try:
            for node in nodes:
                if not isinstance(node, TextNode):
                    node = TextNode.from_base_node(node)

                # Ensure metadata has required fields
                metadata = node.metadata.copy()
                metadata["doc_id"] = metadata.get("file_name", hashlib.md5(node.text.encode()).hexdigest())

                node.metadata = metadata
                node.id_ = self._generate_node_id(node)
                processed_nodes.append(node)

            # Get embeddings in batch
            texts = [node.text for node in processed_nodes]
            try:
                embeddings = self.embed_model.get_text_embedding_batch(texts)
                self.logger.info("Used primary embedding model")
            except Exception as e:
                self.logger.warning(f"Primary embedding failed, using fallback: {e}")
                embeddings = self.fallback_embed_model.get_text_embedding_batch(texts)

            # Assign embeddings
            for node, embedding in zip(processed_nodes, embeddings):
                node.embedding = embedding if isinstance(embedding, list) else embedding.tolist()

            return processed_nodes

        except Exception as e:
            self.logger.error(f"Error processing nodes: {e}")
            raise

    def _generate_node_id(self, node: BaseNode) -> str:
        """Generate consistent node ID"""
        return hashlib.sha256(f"{node.text}{node.metadata}".encode()).hexdigest()