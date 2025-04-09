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
                    "type": "document",
                    "answer": "No relevant information found in knowledge base.",
                    "sources": [],
                    "agent": self.name
                }

            # Step 2: Build synthetic response context from top nodes
            top_node = nodes[0]
            context = top_node.node.text[:3000]  # Truncate to avoid token overflow
            sources = [
                {
                    "text": node.node.text[:300],  # Optional preview
                    "score": node.score,
                    "metadata": node.node.metadata
                } for node in nodes
            ]

            print(f"Context -{context}")
            print(f"Query - {query_str}")

            # Step 3: Build prompt
            prompt = f"""You are a helpful assistant answering questions based on the provided context. 
            Use the information in the context to guide your response as accurately as possible. 
            If the context doesn't clearly contain the answer, it's okay to say: "No relevant answer found" or explain briefly why the information isn't available.
            Avoid guessing or adding details that aren't supported by the context."
            
            Context:
            {context}
            
            Question: {query_str}
            
            Answer:"""

            print("Sending prompt to LLM...")
            llm_response_obj = settings.llm.complete(prompt)
            llm_response = (
                llm_response_obj.text.strip()
                if hasattr(llm_response_obj, "text")
                else str(llm_response_obj).strip()
            )

            print(f"DocumentAgent received response from query engine - {llm_response}")

            return {
                "type": "document",
                "answer": llm_response if llm_response else "No relevant answer found.",
                "sources": sources,
                # "agent": self.name
            }

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return {
                "answer": "Error processing your query",
                "error": str(e),
                # "agent": self.name
            }