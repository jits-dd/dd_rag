import os
import json
import time
from pymilvus import (
    connections,
    utility,
    Collection,
    DataType,
    MilvusException,
    FieldSchema,
    CollectionSchema
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.schema import BaseNode, TextNode
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timezone
from config.settings import settings

logger = logging.getLogger(__name__)

class MilvusStorage:
    def __init__(self, collection_name: str = None, recreate_collection: bool = False):
        self.collection_name = collection_name or settings.MILVUS_COLLECTION
        self._connect()
        self.vector_store = self._initialize_collection(recreate=recreate_collection)

    def get_vector_store(self) -> MilvusVectorStore:
        """Get vector store with current configuration"""
        return self.vector_store

    def _connect(self):
        """Establish connection with authentication and retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT,
                    user=settings.MILVUS_USER,
                    password=settings.MILVUS_PASSWORD,
                    db_name=settings.MILVUS_DATABASE,
                    secure=settings.MILVUS_SECURE
                )
                logger.info("Connected to Milvus successfully")
                return
            except MilvusException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Milvus connection failed after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Connection attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)

    def _create_collection_schema(self):
        """Create complete schema with all required fields"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=65535),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIM),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="participants", dtype=DataType.JSON),
            FieldSchema(name="discussion_topics", dtype=DataType.JSON),
            FieldSchema(name="key_metrics", dtype=DataType.JSON)
        ]
        return CollectionSchema(fields, description="Investment conversation documents")

    def _initialize_collection(self, recreate: bool = False) -> MilvusVectorStore:
        """Initialize collection with proper field mappings"""
        try:
            # Only drop if explicitly requested
            if recreate and self.collection_name in utility.list_collections():
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped existing collection {self.collection_name}")

            # Create new collection only if it doesn't exist
            if self.collection_name not in utility.list_collections():
                logger.info(f"Creating new collection: {self.collection_name}")
                schema = self._create_collection_schema()
                collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    using="default"
                )
                collection.create_index(
                    field_name="embedding",
                    index_params={
                        "metric_type": "IP",
                        "index_type": "HNSW",
                        "params": {"M": 16, "efConstruction": 200}
                    }
                )

            # Initialize vector store with explicit field mappings
            vector_store = MilvusVectorStore(
                collection_name=self.collection_name,
                dim=settings.EMBEDDING_DIM,
                overwrite=False,
                uri=f"{'https' if settings.MILVUS_SECURE else 'http'}://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
                user=settings.MILVUS_USER if not settings.MILVUS_SECURE else None,
                password=settings.MILVUS_PASSWORD if not settings.MILVUS_SECURE else None,
                token=f"{settings.MILVUS_USER}:{settings.MILVUS_PASSWORD}" if settings.MILVUS_SECURE else None,
                text_field="text",  # Explicitly set text field
                embedding_field="embedding",  # Explicitly set embedding field
                metadata_field="metadata",  # Explicitly set metadata field
                stored_fields=["text", "metadata", "file_name", "title", "summary"]  # Fields to store
            )

            return vector_store

        except Exception as e:
            logger.error(f"Collection setup failed: {e}", exc_info=True)
            raise

    async def store_nodes(self, nodes: List[BaseNode]):
        """Store nodes with proper field mapping"""
        try:
            if not nodes:
                logger.warning("No nodes provided for storage")
                return

            # Prepare data for insertion
            data = []
            for node in nodes:
                if not isinstance(node, TextNode):
                    node = TextNode.from_base_node(node)

                # Ensure required fields exist
                node.metadata.setdefault("file_name", "unknown")
                node.metadata.setdefault("title", "")
                node.metadata.setdefault("summary", "")

                data.append({
                    "id": node.node_id,
                    "text": node.text,
                    "embedding": node.embedding,
                    "metadata": node.metadata,
                    "file_name": node.metadata["file_name"],
                    "title": node.metadata["title"],
                    "summary": node.metadata["summary"],
                    "participants": node.metadata.get("participants"),
                    "discussion_topics": node.metadata.get("discussion_topics"),
                    "key_metrics": node.metadata.get("key_metrics")
                })

            # Insert using raw collection API
            col = Collection(self.collection_name)
            col.load()

            insert_result = col.insert(data)
            col.flush()

            logger.info(f"Inserted {len(insert_result.primary_keys)} entities")

            # Verify insertion
            sample = col.query(
                expr=f"id == '{insert_result.primary_keys[0]}'",
                output_fields=["metadata", "file_name"]
            )
            if sample:
                logger.info(f"Verified metadata storage: {sample[0]['metadata']}")
            else:
                logger.warning("Could not verify metadata storage")

        except Exception as e:
            logger.error(f"Failed to store nodes: {e}", exc_info=True)
            raise

    def diagnose_metadata_storage(self):
        """Diagnose metadata storage issues"""
        try:
            col = Collection(self.collection_name)
            col.load()

            # Get first record
            results = col.query(
                expr="",
                limit=1,
                output_fields=["metadata", "file_name"]
            )

            if results:
                logger.info("Diagnostic results:")
                logger.info(f"- File: {results[0]['file_name']}")
                logger.info(f"- Metadata keys: {list(results[0]['metadata'].keys())}")
            else:
                logger.warning("No records found in collection")

        except Exception as e:
            logger.error(f"Diagnostic failed: {e}", exc_info=True)

    def _get_existing_vector_store(self) -> MilvusVectorStore:
        """Get existing vector store instance"""
        return MilvusVectorStore(
            collection_name=self.collection_name,
            dim=settings.EMBEDDING_DIM,
            overwrite=False,
            uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            user=settings.MILVUS_USER,
            password=settings.MILVUS_PASSWORD,
            db_name=settings.MILVUS_DATABASE,
            text_field="text",
            embedding_field="embedding"
        )