import logging
from config.settings import settings
from data_pipeline.loader import AdvancedDocumentLoader
from data_pipeline.processor import DocumentProcessor
from storage.milvus_store import MilvusStorage
from retrieval.retriever import AdvancedRetriever
from orchestrator.query_engine import AdvancedQueryEngine
from app.application import AdvancedRAGApplication
from storage import initialize_milvus

def initialize_system():
    # Initialize Milvus with custom collection
    milvus_storage = initialize_milvus()
    """Initialize all components"""
    # Load and process documents
    loader = AdvancedDocumentLoader()
    processor = DocumentProcessor(settings.embed_model)

    nodes = loader.load_and_chunk()
    processed_nodes = processor.process_nodes(nodes)

    # Store in Milvus with timestamps
    milvus_storage.store_nodes(processed_nodes)
    logging.info(f"Stored {len(processed_nodes)} nodes in Milvus")

    # Create retrieval and query components
    retriever = AdvancedRetriever(milvus_storage.get_vector_store())
    query_engine = AdvancedQueryEngine(retriever)

    # Initialize application
    return AdvancedRAGApplication(query_engine, mode="agent")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        rag_app = initialize_system()
        print("Advanced RAG System Ready. Type 'exit' to quit.")

        while True:
            query = input("\nQuestion: ")
            if query.lower() in ('exit', 'quit'):
                break

            try:
                response = rag_app.query(query, evaluate=True)

                print("\nAnswer:")
                print(response["answer"])

                if response.get("sources"):
                    print("\nSources:")
                    for i, source in enumerate(response["sources"], 1):
                        print(f"{i}. [Score: {source.get('score', 'N/A')}]")
                        print(f"   {source['text'][:200]}...")
                        print(f"   Metadata: {source['metadata']}\n")

                if response.get("evaluation"):
                    eval_data = response["evaluation"]
                    print(f"\nEvaluation Score: {eval_data['score']:.2f}")
                    for metric, result in eval_data['results'].items():
                        print(f"{metric.capitalize()}: {result.feedback}")

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print("Sorry, an error occurred. Please try another question.")

    except Exception as e:
        logger.critical(f"System initialization failed: {e}")
        raise