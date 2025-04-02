from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex
from typing import List
from config.settings import settings

class AdvancedRAGAgent:
    def __init__(self, query_engine, additional_tools: List = None):
        self.query_engine = query_engine
        self.tools = self._setup_tools(additional_tools or [])
        self.agent = self._create_agent()

    def _setup_tools(self, additional_tools):
        base_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="document_retriever",
            description="Access deal conversation notes"
        )
        return [base_tool] + additional_tools

    def _create_agent(self):
        return ReActAgent.from_tools(
            self.tools,
            llm=settings.llm,
            verbose=True,
            max_iterations=6
        )

    def query(self, query_str: str):
        """Execute agent-based query"""
        return self.agent.chat(query_str)