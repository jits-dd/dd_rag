import logging
from config.settings import settings
from data_pipeline.loader import AdvancedConversationLoader
from data_pipeline.processor import ConversationProcessor
from storage.milvus_store import MilvusStorage
from retrieval.retriever import AdvancedConversationRetriever
from orchestrator.query_engine import AdvancedConversationEngine
from app.application import AdvancedRAGApplication
from llama_index.core.schema import BaseNode, QueryBundle
import argparse
import time
from pymilvus import (
    connections, utility, Collection,
    FieldSchema, CollectionSchema, DataType
)

connections.connect(host='localhost', port='19530', user='root', password='Milvus')
col = Collection('conversational_rag')
print(f"\nCollection Status:")
print(f"- Entities: {col.num_entities}")
print(f"- Indexes: {col.indexes}")
print(f"- Schema: {col.schema}")
print(f"- Index: {col.indexes[0].params if col.indexes else 'None'}")

# Sample query to verify data exists
# sample_results = col.query(
#     expr="",
#     output_fields=["text"],
#     limit=3
# )
#
# print("\nSample Documents:")
# for i, doc in enumerate(sample_results):
#     print(f"{i+1}. {doc['text'][:100]}...")

def initialize_milvus():
    """Initialize Milvus with comprehensive validation"""
    try:
        logging.info("Initializing Milvus storage...")
        storage = MilvusStorage()

        # Verify collection is accessible
        col = Collection(settings.MILVUS_COLLECTION)
        col.load()

        if not col.is_empty:
            logging.info(f"Collection '{settings.MILVUS_COLLECTION}' is ready with {col.num_entities} entities")
        else:
            logging.info(f"Collection '{settings.MILVUS_COLLECTION}' is empty")

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

        # Store in Milvus
        milvus_storage.store_nodes(processed_nodes)
        logging.info(f"Stored {len(processed_nodes)} nodes in Milvus")

        # Create retrieval and query components
        retriever = AdvancedConversationRetriever(milvus_storage.get_vector_store(),settings.embed_model)
        query_engine = AdvancedConversationEngine(retriever)

        return AdvancedRAGApplication(query_engine, mode="agent")

    except Exception as e:
        logging.critical(f"System initialization failed: {e}")
        raise

# Test retrieval
queries = [
    "Fusion Food business model",
    "product launches",
    "risk management teams"
]
milvus_storage = initialize_milvus()
retriever = AdvancedConversationRetriever(milvus_storage.get_vector_store(),settings.embed_model)

for query in queries:
    print(f"\nQuery: '{query}'")
    nodes = retriever._retrieve(QueryBundle(query))
    if not nodes:
        print("No results found!")
