from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import PromptTemplate, QueryBundle
from typing import List
from config.settings import settings
import logging

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

DOCUMENT_PROMPT = PromptTemplate("""
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
        self.synthesizer = get_response_synthesizer(
            llm=settings.llm,
            response_mode="compact"
        )



    def query(self, query_str: str):
        """Execute query with proper query bundle"""
        try:
            print(f"\nStarting query processing for: {query_str}")

            # Step 1: Retrieve nodes
            print("Retrieving nodes from vector store...")
            nodes = self.retriever._retrieve(QueryBundle(query_str))
            print(f"Found {len(nodes)} relevant nodes")

            if not nodes:
                return {
                    "answer": "No relevant information found in knowledge base",
                    "sources": []
                }

            print("Generating response...")

            response = self.synthesizer.synthesize(query=query_str, nodes=nodes)
            print(f"QueryEngine query Response -{response}")
            # Step 3: Format response
            return {
                "answer": str(response),
                "sources": [
                    {
                        "text": node.text[:300],
                        "score": node.score,
                        "metadata": node.metadata
                    } for node in response.source_nodes
                ]
            }

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return {
                "answer": "Error processing your query",
                "error": str(e)
            }