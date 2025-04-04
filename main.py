import logging
from config.settings import settings
from data_pipeline.loader import AdvancedConversationLoader
from data_pipeline.processor import ConversationProcessor
from storage.milvus_store import MilvusStorage
from retrieval.retriever import AdvancedConversationRetriever
from orchestrator.query_engine import AdvancedConversationEngine
from app.application import AdvancedRAGApplication
import argparse
import time
from pymilvus import connections, Collection
from llama_index.core import QueryBundle

def initialize_milvus():
    """Initialize Milvus with comprehensive validation"""
    try:
        logging.info("Initializing Milvus storage...")
        storage = MilvusStorage()

        # Verify collection is accessible
        col = Collection(settings.MILVUS_COLLECTION)
        col.load()

        # Debug: Print collection info
        logging.debug(f"Collection schema: {col.schema}")
        logging.debug(f"Number of entities: {col.num_entities}")
        logging.debug(f"Indexes: {col.indexes}")

        if not col.is_empty:
            logging.info(f"Collection '{settings.MILVUS_COLLECTION}' is ready with {col.num_entities} entities")
        else:
            logging.warning(f"Collection '{settings.MILVUS_COLLECTION}' is empty")

        logging.info("Milvus storage initialized successfully")
        return storage

    except Exception as e:
        logging.critical(f"Milvus initialization failed: {e}")
        raise RuntimeError(f"Milvus initialization failed: {e}")

def initialize_system():
    """Initialize all components with robust error handling"""
    try:
        logging.info("Initializing system components...")

        # Initialize storage with retry
        milvus_storage = None
        for attempt in range(3):
            try:
                milvus_storage = initialize_milvus()
                break
            except Exception as e:
                logging.warning(f"Milvus initialization attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        # Load and process documents
        loader = AdvancedConversationLoader()
        processor = ConversationProcessor(settings.embed_model)

        nodes = loader.load_and_chunk()
        logging.info(f"Loaded {len(nodes)} document chunks")

        # Process nodes with retry
        processed_nodes = None
        for attempt in range(3):
            try:
                processed_nodes = processor.process_nodes(nodes)
                break
            except Exception as e:
                logging.warning(f"Processing attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        # Store in Milvus with verification
        milvus_storage.store_nodes(processed_nodes)
        logging.info(f"Stored {len(processed_nodes)} nodes in Milvus")

        # Verify storage
        col = Collection(settings.MILVUS_COLLECTION)
        col.load()
        if col.num_entities != len(processed_nodes):
            logging.error(f"Storage mismatch: Expected {len(processed_nodes)} entities, found {col.num_entities}")
            raise ValueError("Data storage verification failed")

        # Create retrieval and query components
        retriever = AdvancedConversationRetriever(
            milvus_storage.get_vector_store(),
            settings.embed_model
        )
        query_engine = AdvancedConversationEngine(retriever)

        return AdvancedRAGApplication(query_engine, mode="agent")

    except Exception as e:
        logging.critical(f"System initialization failed: {e}")
        raise

def main():
    """Main application entry point with enhanced error handling"""
    parser = argparse.ArgumentParser(description="Advanced RAG System for Conversation Analysis")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    try:
        rag_app = initialize_system()
        print("\nAdvanced Conversation Analysis System Ready. Type 'exit' to quit.\n")

        while True:
            try:
                query = input("\nQuestion: ").strip()
                if query.lower() in ('exit', 'quit'):
                    break
                if not query:
                    continue

                response = rag_app.query(query)

                # Enhanced response handling
                print("\nAnswer:")
                if isinstance(response, dict):
                    print(response.get("answer", "No answer provided"))

                    if response.get("sources"):
                        print("\nSources:")
                        for i, source in enumerate(response["sources"], 1):
                            print(f"{i}. [Relevance: {source.get('score', 0):.3f}]")
                            print(f"   Text: {source.get('text', '')[:200]}...")
                            if metadata := source.get('metadata'):
                                print(f"   Metadata: {metadata}")
                            print()
                    elif "answer" in response:
                        print("\nNote: No source references available")
                else:
                    print(str(response))

            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print("\nSorry, I encountered an error. Please try rephrasing your question.")
                if args.debug:
                    print(f"Debug: {str(e)}")

    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        print("\nSystem initialization failed. Please check logs for details.")
        if args.debug:
            raise
        else:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()