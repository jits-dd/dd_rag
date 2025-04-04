from llama_index.core.tools import QueryEngineTool
from typing import Dict, Any
import logging

class ConversationAgent:
    def __init__(self, query_engine):
        self.name = "Conversation Analyst"
        self.description = "Specializes in analyzing and interpreting conversations"
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)
        self.tool = self._create_tool()

    def _create_tool(self):
        return QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="conversation_analyzer",
            description=(
                "Access to analyzed conversation data. Use for all questions about "
                "dialog exchanges between people. Provides contextual understanding."
            )
        )

    def process(self, query: str) -> Dict[str, Any]:
        """Process conversation-specific queries"""
        try:
            response = self.query_engine.query(query)
            return {
                "type": "conversation",
                "answer": str(response),
                "sources": self._extract_sources(response)
            }
        except Exception as e:
            self.logger.error(f"Conversation analysis failed: {e}")
            return {
                "answer": "Error analyzing conversations",
                "error": str(e)
            }

    def _extract_sources(self, response) -> list:
        return [{
            "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
            "score": node.score,
            "metadata": node.metadata
        } for node in getattr(response, 'source_nodes', [])]