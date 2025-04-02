from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from typing import List
from llama_index.core.schema import Document
from config.settings import settings

class AdvancedDocumentLoader:
    def __init__(self, input_dir: str = "data"):
        self.input_dir = input_dir
        self.parser = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=settings.embed_model
        )

    def load_and_chunk(self) -> List[Document]:
        """Load and semantically chunk documents"""
        documents = SimpleDirectoryReader(
            self.input_dir,
            required_exts=[".pdf", ".docx", ".pptx", ".txt"]
        ).load_data()

        return self.parser.get_nodes_from_documents(documents)