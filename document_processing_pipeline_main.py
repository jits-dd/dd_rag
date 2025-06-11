import logging
import asyncio
from storage.milvus_store import MilvusStorage
from data_pipeline.loader import AdvancedDocumentLoader
from typing import List, Dict, Any, Optional
from llama_index.core.schema import Document, TextNode, BaseNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_processing.log"),
        logging.StreamHandler()
    ]
)

class DocumentProcessingPipeline:
    def __init__(
            self,
            input_dir: str = "data",
            collection_name: str = "investment_conversations",  # Changed to more specific name
            parsing_mode: str = "semantic"  # Better for conversations
    ):
        self.input_dir = input_dir
        self.loader = AdvancedDocumentLoader(
            input_dir=input_dir,
            parsing_mode=parsing_mode
        )
        self.storage = MilvusStorage(collection_name=collection_name, recreate_collection=True)
        self.logger = logging.getLogger(__name__)

    async def validate_metadata(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """Ensure all nodes have required metadata"""
        valid_nodes = []
        for node in nodes:
            if not isinstance(node, TextNode):
                continue

            # Ensure required metadata fields exist
            if "file_name" not in node.metadata:
                node.metadata["file_name"] = "unknown"

            # Add document type
            node.metadata["document_type"] = "investment_conversation"

            valid_nodes.append(node)

        return valid_nodes
    async def run_pipeline(self):
        """Complete document processing pipeline"""
        try:
            # 1. Load and process documents
            self.logger.info("Starting document processing...")
            nodes = await self.loader.load_and_process()

            # Validate metadata
            nodes = await self.validate_metadata(nodes)
            self.logger.info(f"Processed {len(nodes)} document chunks")

            # 2. Store in vector database
            self.logger.info("Storing documents in vector database...")
            await self.storage.store_nodes(nodes)

            # Add diagnostic check
            self.storage.diagnose_metadata_storage()

            # Log some metadata stats
            file_names = {n.metadata.get("file_name") for n in nodes}
            self.logger.info(f"Processed {len(file_names)} unique files")

            return {
                "status": "success",
                "processed_nodes": len(nodes),
                "unique_files": len(file_names),
                "collection": self.storage.collection_name
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

async def document_processing_main():
    # Example usage
    pipeline = DocumentProcessingPipeline(
        input_dir="data",
        collection_name="general_docs",
        parsing_mode="semantic"  # Try "hierarchical" for different strategy
    )

    # Run the full pipeline
    await pipeline.run_pipeline()

if __name__ == "__main__":
    asyncio.run(document_processing_main())
