from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import connections, utility, Collection, FieldSchema, DataType, CollectionSchema
from config.settings import settings
import time
import logging

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
                    password=settings.MILVUS_PASSWORD
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

                # Verify the collection has the required schema
                if not self._validate_collection_schema(col):
                    logging.warning("⚠️ Existing collection has invalid schema")
                    raise RuntimeError("Schema validation failed")

                return MilvusVectorStore(
                    collection_name=settings.MILVUS_COLLECTION,
                    overwrite=False,
                    uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
                    user=settings.MILVUS_USER,
                    password=settings.MILVUS_PASSWORD,
                    db_name=settings.MILVUS_DATABASE
                )

            # Create new collection if it doesn't exist
            logging.info(f"🆕 Creating new collection: {settings.MILVUS_COLLECTION}")

            # Define schema for new collection
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=65535
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=settings.EMBEDDING_DIM
                ),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=65535
                ),
                FieldSchema(
                    name="document_id",
                    dtype=DataType.VARCHAR,
                    max_length=65535
                ),
                FieldSchema(
                    name="metadata",
                    dtype=DataType.JSON
                )
            ]

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
                db_name=settings.MILVUS_DATABASE
            )

            # Verify creation
            time.sleep(2)  # Allow time for propagation
            if settings.MILVUS_COLLECTION not in utility.list_collections():
                raise RuntimeError("Collection creation failed")

            logging.info("✅ Collection created successfully")
            return vector_store

        except Exception as e:
            logging.error(f"❌ Collection setup failed: {e}")
            raise RuntimeError(f"Collection initialization error: {e}")

    def _validate_collection_schema(self, collection):
        """Verify the collection has required fields"""
        required_fields = {
            "id": DataType.VARCHAR,
            "embedding": DataType.FLOAT_VECTOR,
            "text": DataType.VARCHAR,
            "metadata": DataType.JSON
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

    def store_nodes(self, nodes):
        """Safe data insertion with validation"""
        try:
            if not self.vector_store:
                raise RuntimeError("Collection not initialized")

            self.vector_store.add(nodes)
            logging.info(f"📥 Stored {len(nodes)} nodes")

            # Verify insertion
            col = Collection(settings.MILVUS_COLLECTION)
            col.load()
            if col.num_entities == 0:
                logging.warning("⚠️ No entities found after insertion")
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