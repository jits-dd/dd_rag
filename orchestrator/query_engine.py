from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from retrieval.retriever import AdvancedRetriever
from retrieval.reranker import AdvancedReranker
from llama_index.core.query_pipeline import QueryPipeline
from typing import Optional
from config.settings import settings

class AdvancedQueryEngine:
    def __init__(self, retriever: AdvancedRetriever, reranker: Optional[AdvancedReranker] = None):
        self.retriever = retriever
        self.reranker = reranker or AdvancedReranker()
        self.query_pipeline = self._build_query_pipeline()

    def _build_query_pipeline(self):
        """Build end-to-end query pipeline with retrieval and synthesis"""
        qp = QueryPipeline()

        # Add modules
        qp.add_modules({
            "retriever": self.retriever.get_retriever(),
            "reranker": self.reranker.reranker,
            "synthesizer": get_response_synthesizer(llm=settings.llm)
        })

        # Connect pipeline
        qp.add_link("retriever", "reranker")
        qp.add_link("reranker", "synthesizer")

        return qp

    def query(self, query_str: str):
        """Execute the full RAG pipeline"""
        return self.query_pipeline.run(query=query_str)