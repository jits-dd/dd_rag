from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import PromptTemplate
from typing import List, Dict
from llama_index.core.schema import NodeWithScore
from config.settings import settings
import logging

# More focused prompts
CONVERSATION_PROMPT = PromptTemplate("""
You are analyzing stored conversations. The user asked:
{query_str}

Relevant conversation excerpts:
{context_str}

### Instructions ###
1. Directly answer using ONLY the provided conversations
2. Identify speakers and key exchanges
3. Never invent or add information not in the excerpts
4. If unsure, say "The conversations don't contain this information"

Answer:
""")

STANDARD_PROMPT = PromptTemplate("""
User question:
{query_str}

Relevant documents:
{context_str}

### Instructions ###
1. Answer concisely using ONLY the provided documents
2. Cite which document each fact comes from
3. If the answer isn't in the documents, say so

Answer:
""")

class AdvancedConversationEngine:
    def __init__(self, retriever):
        self.retriever = retriever
        self.logger = logging.getLogger(__name__)
        self.response_synthesizer = get_response_synthesizer(
            llm=settings.llm,
            response_mode="compact",
            text_qa_template=CONVERSATION_PROMPT
        )
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=self.response_synthesizer,
            node_postprocessors=[]
        )

    def _select_prompt(self, nodes: List[NodeWithScore]) -> PromptTemplate:
        """Select prompt based on content type"""
        has_conversation = any(
            n.metadata.get("is_conversation", False)
            for n in nodes
        )
        return CONVERSATION_PROMPT if has_conversation else STANDARD_PROMPT

    def query(self, query_str: str) -> Dict[str, any]:
        """Execute query with strict document enforcement"""
        try:
            # Retrieve nodes first
            nodes = self.retriever.retrieve(query_str)
            if not nodes:
                return {
                    "answer": "No relevant information found in stored documents.",
                    "sources": []
                }

            # Update prompt based on content
            prompt = self._select_prompt(nodes)
            self.response_synthesizer.update_prompts(
                {"text_qa_template": prompt}
            )

            # Execute query
            response = self.query_engine.query(query_str)

            # Format response with sources
            sources = []
            for i, node in enumerate(response.source_nodes, 1):
                sources.append({
                    "text": node.text[:300] + "..." if len(node.text) > 300 else node.text,
                    "metadata": node.metadata,
                    "score": node.score,
                    "rank": i
                })

            return {
                "answer": str(response),
                "sources": sources,
                "metadata": {
                    "retrieved_nodes": len(nodes),
                    "prompt_used": prompt.template[:100] + "..." if len(prompt.template) > 100 else prompt.template
                }
            }

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return {
                "answer": "Error processing your query. Please try again.",
                "error": str(e)
            }