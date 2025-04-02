from .milvus_store import MilvusStorage
import logging
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from config.settings import settings

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