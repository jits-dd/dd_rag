from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import connections, utility, Collection, DataType
from config.settings import settings
import logging
from typing import List, Optional
from llama_index.core.schema import BaseNode

class MilvusStorage:
    def __init__(self):
        self._connect()
        self.vector_store = self._initialize_collection()

    def _connect(self):
        """Establish connection with authentication"""
        try:
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT,
                user=settings.MILVUS_USER,
                password=settings.MILVUS_PASSWORD,
                db_name=settings.MILVUS_DATABASE
            )
            logging.info("Connected to Milvus successfully")
        except Exception as e:
            logging.error(f"Milvus connection failed: {e}")
            raise

    def _initialize_collection(self) -> MilvusVectorStore:
        """Initialize or reuse Milvus collection with proper schema"""
        try:
            # Drop existing collection if it has schema issues
            if settings.MILVUS_COLLECTION in utility.list_collections():
                utility.drop_collection(settings.MILVUS_COLLECTION)

            logging.info(f"Creating new collection: {settings.MILVUS_COLLECTION}")
            return self._create_vector_store()

        except Exception as e:
            logging.error(f"Collection setup failed: {e}")
            raise

    def _create_vector_store(self) -> MilvusVectorStore:
        """Create Milvus vector store instance with proper schema"""
        return MilvusVectorStore(
            collection_name=settings.MILVUS_COLLECTION,
            dim=settings.EMBEDDING_DIM,
            overwrite=True,
            uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            user=settings.MILVUS_USER,
            password=settings.MILVUS_PASSWORD,
            db_name=settings.MILVUS_DATABASE,
            text_field="text",
            embedding_field="embedding",
            index_params={
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            },
            additional_fields=[
                {
                    "name": "doc_id",
                    "type": DataType.VARCHAR,
                    "params": {"max_length": 256},
                    "default_value": "default_doc",
                    "nullable": True
                }
            ]
        )

    def store_nodes(self, nodes: List[BaseNode]):
        """Store nodes with proper doc_id handling"""
        try:
            # Prepare data with doc_id
            data = [{
                "id": node.id_,
                "text": node.get_content(),
                "embedding": node.embedding,
                "metadata": node.metadata,
                "doc_id": node.metadata.get("doc_id", "default_doc")
            } for node in nodes]

            # Insert data
            self.vector_store.client.insert(
                self.vector_store.collection_name,
                data
            )

            # Ensure persistence
            col = Collection(self.vector_store.collection_name)
            col.flush()

            logging.info(f"Stored {len(nodes)} nodes successfully")

        except Exception as e:
            logging.error(f"Failed to store nodes: {e}")
            raise

    def get_vector_store(self) -> MilvusVectorStore:
        return type('VectorStoreConfig', (), {
            'collection_name': settings.MILVUS_COLLECTION,
            'embedding_field': 'embedding',
            'text_field': 'text'
        })()



class CustomMilvusVectorStore(MilvusVectorStore):
    def query(
            self,
            query_embedding: List[float],
            similarity_top_k: int,
            **kwargs
    ) -> List[dict]:
        """Properly formatted query method for Milvus"""
        print("\nInside milvus_store query method")
        print(f"Query embedding dim: {len(query_embedding)}")
        print(f"Similarity top k: {similarity_top_k}")

        try:
            # Ensure collection is loaded
            col = Collection(self.collection_name)
            col.load()

            # Prepare search parameters
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }

            # Execute search
            results = col.search(
                data=[query_embedding],
                anns_field=self.embedding_field,
                param=search_params,
                limit=similarity_top_k,
                output_fields=[self.text_field, "metadata"]
            )

            # Format results
            formatted_results = [
                {
                    "text": hit.entity.get(self.text_field),
                    "metadata": hit.entity.get("metadata"),
                    "embedding": hit.entity.get(self.embedding_field),
                    "score": hit.score
                }
                for hit in results[0]
            ]

            print(f"Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logging.error(f"Milvus query failed: {e}")
            return []