import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import logging
from pymilvus import DataType
from config.app_config import config

class Settings:
    # Embedding Models
    EMBEDDING_MODEL = "text-embedding-3-large"
    HF_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM = 3072  # For text-embedding-3-large

    # Milvus Configuration
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_USER = os.getenv("MILVUS_USER", "root")
    MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "Milvus")
    MILVUS_DATABASE = os.getenv("MILVUS_DATABASE", "default")
    MILVUS_COLLECTION = "general_docs"
    MILVUS_SECURE: bool = False

    # Data Processing
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 200
    CONVERSATION_WINDOW = 4

    # Retrieval
    RETRIEVAL_TOP_K = 5
    RERANK_TOP_K = 3

    @property
    def embed_model(self):
        """Get embedding model with fallback"""
        try:
            return OpenAIEmbedding(
                model=self.EMBEDDING_MODEL,
                api_key=config['llm_model']['api_key'],
                timeout=30,  # Increased timeout
                max_retries=3
            )
        except Exception as e:
            logging.warning(f"OpenAI embedding failed, using HuggingFace: {e}")
            return HuggingFaceEmbedding(
                model_name=self.HF_EMBEDDING_MODEL
            )

    @property
    def llm(self):
        return OpenAI(
            model="gpt-4o-2024-05-13",
            temperature=0.1,
            max_tokens=2000,
            api_key=config['llm_model']['api_key']
        )

settings = Settings()