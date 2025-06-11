import logging
from config.settings import settings
from agents.conversation_agent import ConversationAgent
from agents.document_agent import DocumentAgent
from orchestrator.orchestrator import AgentOrchestrator
import argparse
import asyncio
from storage.milvus_store import MilvusStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("query_engine.log"),
        logging.StreamHandler()
    ]
)

def initialize_query_system():
    """Initialize only the query components"""
    try:
        # Initialize Milvus connection and get vector store
        storage = MilvusStorage(recreate_collection=False)
        vector_store = storage.vector_store  # Get the vector store instance

        # Initialize retriever with the vector store
        from retrieval.retriever import AdvancedConversationRetriever
        from orchestrator.query_engine import AdvancedConversationEngine

        retriever = AdvancedConversationRetriever(vector_store, settings.embed_model)
        print(f"Retriever main {retriever}")
        query_engine = AdvancedConversationEngine(retriever)
        print(f"Query Engine main {query_engine}")

        # Initialize agents
        agents = [
            ConversationAgent(query_engine),
            DocumentAgent(query_engine)
        ]

        return AgentOrchestrator(agents)

    except Exception as e:
        logging.error(f"Query system initialization failed: {e}")
        raise

def display_response(response):
    """Improved response display"""
    print("\n=== Response ===")
    print(response.get("answer", "No answer could be generated"))

    if "sources" in response and response["sources"]:
        print("\nSources:")
        for i, source in enumerate(response["sources"], 1):
            print(f"{i}. [Relevance: {source.get('score', 0):.2f}]")
            if 'title' in source.get('metadata', {}):
                print(f"   Title: {source['metadata']['title']}")
            print(f"   Excerpt: {source.get('text', '')[:200]}...")
    print("")

async def main():
    """Query-only entry point"""
    parser = argparse.ArgumentParser(description="Query Engine for Investment Conversations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize only query components
        orchestrator = initialize_query_system()
        logging.info("Query system initialized successfully")

        # Interaction loop
        print("\nInvestment Conversation Query System")
        print("Type 'exit' to quit\n")
        while True:
            try:
                query = input("\nYour question: ").strip()
                if query.lower() in ('exit', 'quit'):
                    break
                if not query:
                    continue

                # Process query
                response = orchestrator.orchestrate_task(query)
                display_response(response)

            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                break
            except Exception as e:
                logging.error(f"Query processing error: {e}")
                print("\nSorry, I encountered an error processing your question")

    except Exception as e:
        logging.critical(f"System startup failed: {e}")
        print("\nSystem failed to initialize. Please check logs.")

if __name__ == "__main__":
    asyncio.run(main())