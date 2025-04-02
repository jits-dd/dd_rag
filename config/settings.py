import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.constants import DEFAULT_EMBEDDING_DIM

class Settings:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-ada-002"
    LLM_MODEL = "gpt-3.5-turbo"

    # Milvus
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_USER = os.getenv("MILVUS_USER", "root")
    MILVUS_PASSWORD = os.getenv("MILVUS_USER", "Milvus")
    MILVUS_DATABASE = os.getenv("MILVUS_DATABASE", "default")
    MILVUS_COLLECTION = "advanced_rag_test"
    EMBEDDING_DIM = 1536  # 1536 for ada-002

    # Data Processing
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 200

    @property
    def embed_model(self):
        return OpenAIEmbedding(
            model=self.EMBEDDING_MODEL,
            api_key=self.OPENAI_API_KEY
        )

    @property
    def llm(self):
        return OpenAI(
            model=self.LLM_MODEL,
            temperature=0.1,
            max_tokens=2000,
            api_key=self.OPENAI_API_KEY
        )

settings = Settings()