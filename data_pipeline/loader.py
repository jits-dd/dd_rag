from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.schema import Document, TextNode
from typing import List
import logging
import re

class AdvancedConversationLoader:
    def __init__(self, input_dir: str = "data"):
        self.input_dir = input_dir
        self.logger = logging.getLogger(__name__)

    def _parse_conversation(self, text: str, file_name: str) -> List[Document]:
        """Parse conversation text into structured dialog turns"""
        dialog_pattern = re.compile(r"(?:\n|^)(\w+:\s*|\[.*?\]\s*)(.*?)(?=\n\w+:\s*|\n\[.*?\]\s*|\Z)", re.DOTALL)
        matches = dialog_pattern.findall(text)

        documents = []
        current_chunk = []

        for speaker, content in matches:
            content = content.strip()
            if not content:
                continue

            current_chunk.append(f"{speaker.strip()}: {content}")

            if len(current_chunk) >= settings.CONVERSATION_WINDOW:
                documents.append(Document(
                    text="\n".join(current_chunk),
                    metadata={
                        "source": "conversation",
                        "is_conversation": True,
                        "file_name": file_name
                    }
                ))
                current_chunk = []

        if current_chunk:
            documents.append(Document(
                text="\n".join(current_chunk),
                metadata={
                    "source": "conversation",
                    "is_conversation": True,
                    "file_name": file_name
                }
            ))

        return documents

    def load_and_chunk(self) -> List[TextNode]:
        """Load and semantically chunk documents"""
        try:
            raw_docs = SimpleDirectoryReader(self.input_dir).load_data()
            all_docs = []

            for doc in raw_docs:
                if "conversation" in doc.metadata.get("file_name", "").lower():
                    all_docs.extend(self._parse_conversation(doc.text, doc.metadata.get("file_name", "")))
                else:
                    all_docs.append(doc)

            node_parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[1024, 512, 256]
            )
            nodes = node_parser.get_nodes_from_documents(all_docs)
            return nodes  # Return all nodes, not just leaf nodes

        except Exception as e:
            self.logger.error(f"Error loading documents: {e}")
            raise