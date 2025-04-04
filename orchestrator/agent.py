from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex
from typing import List, Optional, Dict
from config.settings import settings
import logging

class AdvancedConversationAgent:
    def __init__(self, query_engine, additional_tools: Optional[List] = None):
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)
        self.tools = self._setup_tools(additional_tools or [])
        self.agent = self._create_agent()

    def _setup_tools(self, additional_tools):
        """Setup conversation-specific tools with enforced document usage"""
        base_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="conversation_analyzer",
            description=(
                "Access to analyzed conversation data. Use this when you need to "
                "reference or analyze conversations between people. Provides "
                "contextual understanding of dialog exchanges. MUST use this tool "
                "for any factual questions about conversations."
            ),
            tool_metadata={
                "required": True  # Force usage for conversation queries
            }
        )
        return [base_tool] + additional_tools

    def _create_agent(self):
        """Create agent with strict document usage requirements"""
        return ReActAgent.from_tools(
            self.tools,
            llm=settings.llm,
            verbose=True,
            max_iterations=8,
            system_prompt=(
                "You are a specialized AI for analyzing conversations. Your role is to:"
                "\n1. ALWAYS use the conversation_analyzer tool for any factual questions"
                "\n2. Reference specific dialog turns when answering"
                "\n3. Never invent conversations - only use what's in the knowledge base"
                "\n4. If no relevant conversations are found, say so explicitly"
                "\n\nYou MUST use tools to access conversation data for all factual queries."
            ),
            context=(
                "You are required to use the conversation_analyzer tool for all "
                "questions about conversations. Never make up dialog - only reference "
                "what exists in the stored conversations."
            )
        )

    def query(self, query_str: str):
        try:
            # First get direct retrieval results to verify content exists
            nodes = self.query_engine.retriever.retrieve(query_str)
            if not nodes:
                return {
                    "answer": "No relevant conversations found in the knowledge base.",
                    "details": "I couldn't find any stored conversations related to your query."
                }

            # Now use the agent which will be forced to use the tool
            response = self.agent.chat(query_str)

            # Ensure the response cites sources
            source_nodes = getattr(response, 'source_nodes', [])
            if not source_nodes:
                raise ValueError("Agent response must cite sources")

            return {
                "answer": str(response),
                "sources": self._extract_sources(response),
                "metadata": getattr(response, "metadata", {})
            }

        except Exception as e:
            self.logger.error(f"Agent query failed: {e}")
            return {
                "answer": "I cannot answer that question without reference to specific conversations.",
                "details": "The system requires access to stored conversations to respond accurately."
            }

    def _extract_sources(self, response) -> List[Dict]:
        sources = []
        for i, node in enumerate(getattr(response, "source_nodes", []), 1):
            source_info = {
                "text": node.text[:500] + "..." if len(node.text) > 500 else node.text,
                "metadata": node.metadata,
                "score": node.score if hasattr(node, "score") else None,
                "rank": i
            }
            sources.append(source_info)
        return sources