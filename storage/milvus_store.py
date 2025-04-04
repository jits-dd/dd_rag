from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import (
    connections, utility, Collection,
    FieldSchema, CollectionSchema, DataType
)
from config.settings import settings
import time
import logging
from typing import List
from llama_index.core.schema import BaseNode
import argparse

class MilvusStorage:
    def __init__(self):
        self._connect_with_retry()
        self.vector_store = self._initialize_collection()

    def _connect_with_retry(self, max_retries=3, initial_delay=1):
        """Robust connection handling with authentication"""
        for attempt in range(max_retries):
            try:
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT,
                    user=settings.MILVUS_USER,
                    password=settings.MILVUS_PASSWORD,
                    db_name=settings.MILVUS_DATABASE
                )
                if connections.has_connection("default"):
                    logging.info("✅ Successfully connected to Milvus")
                    return
                time.sleep(initial_delay * (2 ** attempt))
            except Exception as e:
                logging.warning(f"⚠️ Connection attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to connect after {max_retries} attempts")

    def _initialize_collection(self):
        """Initialize collection with proper schema handling"""
        try:
            # Check if collection exists
            if settings.MILVUS_COLLECTION in utility.list_collections():
                logging.info(f"🔄 Using existing collection: {settings.MILVUS_COLLECTION}")
                col = Collection(settings.MILVUS_COLLECTION)
                col.load()

                # if not self._validate_collection_schema(col):
                #     logging.warning("⚠️ Existing collection has invalid schema - recreating")
                #     utility.drop_collection(settings.MILVUS_COLLECTION)
                #     return self._create_new_collection()

                return self._get_vector_store(existing=True)

            return self._create_new_collection()

        except Exception as e:
            logging.error(f"❌ Collection setup failed: {e}")
            raise RuntimeError(f"Collection initialization error: {e}")

    def _create_new_collection(self):
        """Create new collection with correct schema"""
        logging.info(f"🆕 Creating new collection: {settings.MILVUS_COLLECTION}")

        vector_store = MilvusVectorStore(
            collection_name=settings.MILVUS_COLLECTION,
            dim=settings.EMBEDDING_DIM,
            overwrite=True,
            index_params={
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            },
            uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            user=settings.MILVUS_USER,
            password=settings.MILVUS_PASSWORD,
            db_name=settings.MILVUS_DATABASE,
            # These fields must match what Milvus actually creates
            text_field="text",
            id_field="id",
            metadata_field="metadata",
        )

        # Verify creation
        time.sleep(2)
        if settings.MILVUS_COLLECTION not in utility.list_collections():
            raise RuntimeError("Collection creation failed")

        logging.info("✅ Collection created successfully")
        return vector_store

    def _get_vector_store(self, existing=False):
        """Get vector store with proper configuration"""
        vector_store = MilvusVectorStore(
            collection_name=settings.MILVUS_COLLECTION,
            dim=3072,  # Must match your embedding dimension
            overwrite=True,
            index_params={
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            },
            text_field="text",  # Must match your schema
            embedding_field="embedding"  # Must match your vector field
        )
        return vector_store
        # return MilvusVectorStore(
        #     collection_name=settings.MILVUS_COLLECTION,
        #     overwrite=not existing,
        #     uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        #     user=settings.MILVUS_USER,
        #     password=settings.MILVUS_PASSWORD,
        #     db_name=settings.MILVUS_DATABASE,
        #     text_field="text",
        #     id_field="id",
        #     metadata_field="metadata",
        # )

    def _validate_collection_schema(self, collection):
        """Verify the collection has required fields"""
        required_fields = {
            "id": DataType.VARCHAR,
            "doc_id": DataType.VARCHAR,
            "text": DataType.VARCHAR,
            "embedding": DataType.FLOAT_VECTOR
        }

        schema = collection.schema
        for field_name, field_type in required_fields.items():
            if field_name not in schema.fields:
                logging.error(f"Missing required field: {field_name}")
                return False
            if schema.fields[field_name].dtype != field_type:
                logging.error(f"Invalid type for field {field_name}")
                return False
        return True

    # In milvus_store.py - modify the store_nodes method
    def store_nodes(self, nodes: List[BaseNode]):
        # parser = argparse.ArgumentParser(description="Advanced RAG System for Conversation Analysis")
        # parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        # args = parser.parse_args()
        # # Configure logging
        # logging.basicConfig(
        #     level=logging.DEBUG if args.debug else logging.INFO,
        #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        # )
        # logger = logging.getLogger(__name__)
        """Safe data insertion with validation"""
        try:
            if not self.vector_store:
                raise RuntimeError("Collection not initialized")

            # Convert nodes to proper format
            data = []
            for node in nodes:
                data.append({
                    "id": node.id_,
                    "text": node.get_content(),
                    "embedding": node.embedding,
                    "metadata": node.metadata,
                    "doc_id": node.metadata.get("doc_id", "")
                })

            # Insert and flush to ensure persistence
            self.vector_store.client.insert(self.vector_store.collection_name, data)
            # New: Proper flush implementation
            try:
                # For Milvus 2.x
                self.vector_store.client.flush([self.vector_store.collection_name])
            except Exception as flush_error:
                logging.warning(f"Flush failed, trying alternative: {flush_error}")
                # Alternative flush method
                col = Collection(settings.MILVUS_COLLECTION)
                col.flush()

            # Verify with a direct count query
            col = Collection(self.vector_store.collection_name)
            col.load()
            actual_count = col.num_entities
            logging.info(f"✅ Verified {actual_count} entities in collection")

            if actual_count != len(nodes):
                logging.error(f"❌ Mismatch: tried to insert {len(nodes)} but collection has {actual_count}")

        except Exception as e:
            logging.error(f"❌ Data storage failed: {e}")
            raise

    def get_vector_store(self):
        """Get verified vector store instance"""
        if not self.vector_store:
            raise RuntimeError("Collection not available")
        return self.vector_store

    def __del__(self):
        """Clean up connections"""
        try:
            connections.disconnect("default")
        except:
            pass