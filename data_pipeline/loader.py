from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceWindowNodeParser,
    HierarchicalNodeParser,
    get_leaf_nodes
)
from typing import List, Optional
from llama_index.core.schema import Document, TextNode
from config.settings import settings
import logging
import re

class AdvancedConversationLoader:
    def __init__(self, input_dir: str = "data"):
        self.input_dir = input_dir
        self.logger = logging.getLogger(__name__)

    def _parse_conversation(self, text: str) -> List[Document]:
        """Parse conversation text into structured dialog turns"""
        # Split by common dialog patterns
        dialog_pattern = re.compile(r"(?:\n|^)(\w+:\s*|\[.*?\]\s*)(.*?)(?=\n\w+:\s*|\n\[.*?\]\s*|\Z)", re.DOTALL)
        matches = dialog_pattern.findall(text)

        documents = []
        current_speaker = None
        current_chunk = []

        for speaker, content in matches:
            content = content.strip()
            if not content:
                continue

            # Group turns by conversation window
            if len(current_chunk) >= settings.CONVERSATION_WINDOW:
                doc = Document(
                    text="\n".join(current_chunk),
                    metadata={
                        "source": "conversation",
                        "speakers": {current_speaker} if current_speaker else "unknown"
                    }
                )
                documents.append(doc)
                current_chunk = []

            current_chunk.append(f"{speaker.strip()}: {content}")
            current_speaker = speaker.split(':')[0] if ':' in speaker else None

        if current_chunk:
            doc = Document(
                text="\n".join(current_chunk),
                metadata={
                    "source": "conversation",
                    "speakers": {current_speaker} if current_speaker else "unknown"
                }
            )
            documents.append(doc)

        return documents

    def load_and_chunk(self) -> List[Document]:
        """Load and semantically chunk conversation documents"""
        try:
            # Load raw documents
            raw_docs = SimpleDirectoryReader(
                self.input_dir,
                required_exts=[".txt", ".md", ".csv"],
                file_metadata=lambda x: {"file_name": x}
            ).load_data()

            # Process conversations
            all_docs = []
            for doc in raw_docs:
                if "conversation" in doc.metadata.get("file_name", "").lower():
                    conversation_docs = self._parse_conversation(doc.text)
                    all_docs.extend(conversation_docs)
                else:
                    all_docs.append(doc)

            # Use hierarchical chunking for better context
            node_parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[1024, 512, 256],
                include_metadata=True
            )

            nodes = node_parser.get_nodes_from_documents(all_docs)
            leaf_nodes = get_leaf_nodes(nodes)

            self.logger.info(f"Loaded {len(leaf_nodes)} conversation chunks")
            return leaf_nodes

        except Exception as e:
            self.logger.error(f"Error loading documents: {e}")
            raise