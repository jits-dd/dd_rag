from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import PromptTemplate, QueryBundle
from typing import List
from config.settings import settings
import logging

CONVERSATION_PROMPT = PromptTemplate("""
You are an analytical assistant reviewing stored conversations to answer the user's question.

User's Question:
{query_str}

Relevant Conversation Excerpts:
{context_str}

### Instructions ###
1. Carefully read the entire conversation excerpts before answering.
2. Base your answer strictly on the provided conversations. Do NOT make assumptions or invent information.
3. Clearly identify the speaker(s) (e.g., User, Agent) and reference specific exchanges when needed.
4. If the information is not clearly present in the excerpts, respond with: "The conversations don't contain this information."

Answer:
""")

DOCUMENT_PROMPT = PromptTemplate("""
You are a precise assistant helping answer the user's question based only on the provided documents.

User's Question:
{query_str}

Relevant Documents:
{context_str}

### Instructions ###
1. Read all documents carefully before answering.
2. Provide a concise answer using only the information from the documents.
3. Clearly cite which document (or part) each fact comes from.
4. If the answer is not explicitly stated in the documents, respond with: "The documents do not contain this information."

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

    def _select_prompt(self, nodes: List) -> PromptTemplate:
        """Select prompt based on content type"""
        if any(n.metadata.get("is_conversation", False) for n in nodes):
            return CONVERSATION_PROMPT
        return DOCUMENT_PROMPT

    def query(self, query_str: str):
        """Execute query with proper query bundle"""
        try:
            print(f"\nStarting query processing for: {query_str}")

            # Step 1: Retrieve nodes
            # print("Retrieving nodes from vector store...")
            # nodes = self.retriever._retrieve(QueryBundle(query_str))
            # print(f"Found {len(nodes)} relevant nodes")
            #
            # if not nodes:
            #     return {
            #         "answer": "No relevant information found in knowledge base",
            #         "sources": []
            #     }
            #
            print("Generating response...")
            query_engine = RetrieverQueryEngine(
                retriever=self.retriever,
                response_synthesizer=self.synthesizer
            )
            response = query_engine.query(query_str)
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

    # def query(self, query_str: str):
    #     try:
    #         query_bundle = QueryBundle(query_str)
    #         nodes = self.retriever.retrieve(query_bundle)
    #
    #         if not nodes:
    #             return {
    #                 "answer": "No relevant information found in our records",
    #                 "sources": []
    #             }
    #
    #         # Get the most relevant node for context
    #         context = nodes[0].node.text
    #         metadata = nodes[0].node.metadata
    #
    #         prompt = f"""Based on this context:
    #     {context}
    #
    #     Question: {query_str}
    #
    #     Provide a concise answer focusing on the business model aspects:"""
    #
    #         llm_response = settings.llm.complete(prompt)
    #
    #         return {
    #             "answer": str(llm_response),
    #             "sources": [{
    #                 "text": nodes[0].node.text,
    #                 "score": nodes[0].score,
    #                 "metadata": metadata
    #             }]
    #         }
    #
    #     except Exception as e:
    #         self.logger.error(f"Query failed: {e}")
    #         return {
    #             "answer": "Error processing your query",
    #             "error": str(e)
    #         }