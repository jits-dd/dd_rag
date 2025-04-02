from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.openai import OpenAI
from typing import List
from llama_index.core.schema import NodeWithScore
from config.settings import settings

class AdvancedReranker:
    def __init__(self):
        self.reranker = LLMRerank(
            llm=settings.llm,
            top_n=5  # return top 3 after reranking
        )

    def rerank(self, query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        return self.reranker.postprocess_nodes(nodes, query_str=query)