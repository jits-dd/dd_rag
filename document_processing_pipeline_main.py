import logging
import asyncio
from storage.milvus_store import MilvusStorage
from data_pipeline.loader import AdvancedDocumentLoader
from typing import List, Dict, Any, Optional

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
            collection_name: str = "documents",
            parsing_mode: str = "hierarchical"  # or "semantic"
    ):
        self.input_dir = input_dir
        self.loader = AdvancedDocumentLoader(
            input_dir=input_dir,
            parsing_mode=parsing_mode
        )
        self.storage = MilvusStorage(collection_name=collection_name)

    async def run_pipeline(self):
        """Complete document processing pipeline"""
        try:
            # 1. Load and process documents
            logging.info("Starting document processing...")
            nodes = await self.loader.load_and_process()
            logging.info(f"Processed {len(nodes)} document chunks")

            # 2. Store in vector database
            logging.info("Storing documents in vector database...")
            await self.storage.store_nodes(nodes)
            logging.info("Pipeline completed successfully")

            return {
                "status": "success",
                "processed_nodes": len(nodes),
                "collection": self.storage.collection_name
            }
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
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

