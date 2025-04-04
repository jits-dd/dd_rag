import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import logging

class Settings:
    # Embedding Models
    EMBEDDING_MODEL = "text-embedding-3-large"
    HF_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

    # Milvus Configuration
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_USER = os.getenv("MILVUS_USER", "root")
    MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "Milvus")
    MILVUS_DATABASE = os.getenv("MILVUS_DATABASE", "default")
    MILVUS_COLLECTION = "conversational_rag"
    EMBEDDING_DIM = 3072  # For text-embedding-3-large

    # Data Processing
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 200
    CONVERSATION_WINDOW = 3

    # Retrieval
    RETRIEVAL_TOP_K = 7
    RERANK_TOP_K = 3

    # Evaluation settings
    ENABLE_EVALUATION = False

    @property
    def embed_model(self):
        """Get embedding model with robust fallback"""
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            logging.warning("OPENAI_API_KEY not found, using HuggingFace embeddings")
            return self._get_hf_embed_model()

        try:
            # Test OpenAI connection with a simple request
            test_embed = OpenAIEmbedding(
                model=self.EMBEDDING_MODEL,
                api_key=openai_key,
                timeout=10
            )
            # Small test request
            test_embed.get_text_embedding("test")
            return test_embed
        except Exception as e:
            logging.warning(f"OpenAI embedding failed, falling back to HuggingFace: {e}")
            return self._get_hf_embed_model()

    def _get_hf_embed_model(self):
        """Get HuggingFace embedding model"""
        return HuggingFaceEmbedding(
            model_name=self.HF_EMBEDDING_MODEL,
            device="cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"
        )

    @property
    def llm(self):
        return OpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            max_tokens=2000,
            api_key=os.getenv("OPENAI_API_KEY")
        )

settings = Settings()