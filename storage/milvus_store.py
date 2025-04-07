import os
import time
from pymilvus import (
    connections,
    utility,
    Collection,
    DataType,
    MilvusException
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.schema import BaseNode
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timezone
from config.settings import settings
from llama_index.embeddings.openai import OpenAIEmbedding

class MilvusStorage:
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.MILVUS_COLLECTION
        self._connect()
        self.vector_store = self._initialize_collection()

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
                logging.info("Connected to Milvus successfully")
                return
            except MilvusException as e:
                if attempt == max_retries - 1:
                    logging.error(f"Milvus connection failed after {max_retries} attempts: {e}")
                    raise
                logging.warning(f"Connection attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff

    def _initialize_collection(self) -> MilvusVectorStore:
        """Initialize collection with flexible schema"""
        try:
            if self.collection_name in utility.list_collections():
                col = Collection(self.collection_name)
                if not self._validate_schema(col.schema):
                    utility.drop_collection(self.collection_name)
                    logging.info("Recreated collection due to schema mismatch")
                    return self._create_vector_store()
                return self._get_existing_vector_store()

            logging.info(f"Creating new collection: {self.collection_name}")
            return self._create_vector_store()

        except Exception as e:
            logging.error(f"Collection setup failed: {e}")
            raise

    def _validate_schema(self, schema) -> bool:
        """Flexible schema validation"""
        required_fields = {
            "text": DataType.VARCHAR,
            "embedding": DataType.FLOAT_VECTOR
        }
        optional_fields = {
            "metadata": DataType.JSON,
            "title": DataType.VARCHAR,
            "summary": DataType.VARCHAR
        }

        has_required = all(
            any(f.name == field and f.dtype == dtype
                for f in schema.fields)
            for field, dtype in required_fields.items()
        )

        return has_required

    def _create_vector_store(self) -> MilvusVectorStore:
        """Create vector store with flexible schema options"""
        return MilvusVectorStore(
            collection_name=self.collection_name,
            dim=settings.EMBEDDING_DIM,
            overwrite=True,
            uri=f"{'https' if settings.MILVUS_SECURE else 'http'}://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            token=f"{settings.MILVUS_USER}:{settings.MILVUS_PASSWORD}" if settings.MILVUS_SECURE else None,
            user=settings.MILVUS_USER if not settings.MILVUS_SECURE else None,
            password=settings.MILVUS_PASSWORD if not settings.MILVUS_SECURE else None,
            db_name=settings.MILVUS_DATABASE,
            text_field="text",
            embedding_field="embedding",
            index_params={
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            },
            search_params={"metric_type": "IP", "params": {"ef": 50}},
            consistency_level="Strong",
            additional_fields=[
                {
                    "name": "metadata",
                    "type": DataType.JSON,
                    "description": "Document metadata",
                    "nullable": True
                },
                {
                    "name": "title",
                    "type": DataType.VARCHAR,
                    "params": {"max_length": 512},
                    "description": "Document title",
                    "nullable": True
                },
                {
                    "name": "summary",
                    "type": DataType.VARCHAR,
                    "params": {"max_length": 2048},
                    "description": "Document summary",
                    "nullable": True
                }
            ]
        )

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

    async def store_nodes(self, nodes: List[BaseNode]):
        """Store nodes with flexible metadata handling"""
        try:
            # Prepare data for insertion
            data = []
            for node in nodes:
                if not hasattr(node, 'embedding') or not node.embedding:
                    continue

                node.metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
                data.append(node)

                # # Prepare document data
                # doc_data = {
                #     "text": node.get_content(),
                #     "embedding": node.embedding,
                #     "metadata": node.metadata,
                #     "title": node.metadata.get("title", ""),
                #     "summary": node.metadata.get("summary", ""),
                #     "updated_at": datetime.now(timezone.utc).isoformat()
                # }

            if not data:
                logging.warning("No valid nodes to store")
                return

            # Insert data with batch processing
            batch_size = 100
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                self.vector_store.add(batch)
                logging.info(f"Inserted batch {i//batch_size + 1}")

            # Create index if not exists
            col = Collection(self.collection_name)
            if not col.has_index():
                col.create_index(
                    field_name="embedding",
                    index_params={
                        "metric_type": "IP",
                        "index_type": "HNSW",
                        "params": {"M": 16, "efConstruction": 200}
                    }
                )
                logging.info("Created new index")

            col.flush()
            logging.info(f"Stored {len(data)} nodes successfully")

        except Exception as e:
            logging.error(f"Failed to store nodes: {e}")
            raise

    def get_vector_store(self) -> MilvusVectorStore:
        """Get vector store with current configuration"""
        return self.vector_store

class CustomMilvusVectorStore(MilvusVectorStore):
    def query(
            self,
            query_embedding: List[float],
            similarity_top_k: int,
            filters: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """Enhanced query method with filtering support"""
        try:
            col = Collection(self.collection_name)
            col.load()

            # Prepare search parameters
            search_params = {
                "metric_type": "IP",
                "params": {"ef": 50}  # Search range for HNSW
            }

            # Prepare filter expression if provided
            expr = None
            if filters:
                filter_parts = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        filter_parts.append(f"{key} in {value}")
                    else:
                        filter_parts.append(f"{key} == '{value}'")
                expr = " and ".join(filter_parts)

            # Execute search
            results = col.search(
                data=[query_embedding],
                anns_field=self.embedding_field,
                param=search_params,
                limit=similarity_top_k,
                expr=expr,
                output_fields=["text", "metadata", "title", "summary"]
            )

            # Format results with enhanced information
            formatted_results = []
            for hit in results[0]:
                result = {
                    "text": hit.entity.get("text"),
                    "metadata": hit.entity.get("metadata", {}),
                    "title": hit.entity.get("title", ""),
                    "summary": hit.entity.get("summary", ""),
                    "score": hit.score,
                    "id": hit.id
                }
                formatted_results.append(result)

            logging.info(f"Retrieved {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logging.error(f"Milvus query failed: {e}")
            return []