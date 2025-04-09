from llama_index.core.tools import QueryEngineTool
from typing import Dict, Any
import logging
from config.settings import settings

class DocumentAgent:
    def __init__(self, query_engine):
        self.name = "Document Analyst"
        self.description = "Specializes in analyzing document content"
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)
        self.tool = self._create_tool()

    def _create_tool(self):
        return QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="document_analyzer",
            description=(
                "Access to analyzed document content. Use for factual questions "
                "about documents, reports, or general knowledge."
            )
        )

    def process(self, query: str) -> Dict[str, Any]:
        """Process document query with proper error handling"""
        try:
            print(f"\nDocumentAgent processing query: {query}")
            response = self.query_engine.query(query)
            if not response:
                raise ValueError("Empty response from query engine")

            print(f"DocumentAgent received response from query engine - {response}")
            # Extract answer and sources from response
            context = response.get("answer", "No answer found")
            sources = response.get("sources", [])

            # Construct the new LLM prompt
            prompt = f"""You are a helpful assistant answering questions strictly based on the provided context. 
            Only answer if the context clearly answers the user's question. 
            Do NOT make assumptions or hallucinate any facts. 
            If the context is irrelevant, unclear, or incomplete, simply respond with: "No relevant answer found."
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:"""

            # Generate refined response using LLM
            llm_response_obj = settings.llm.complete(prompt)
            llm_response = llm_response_obj.text.strip() if hasattr(llm_response_obj, "text") else str(llm_response_obj).strip()

            return {
                "type": "document",
                "answer": llm_response if llm_response else "No relevant answer found.",
                "sources": sources,  # Keeping original sources from retrieval
                "agent": self.name
            }
            # return {
            #     "type": "document",
            #     "answer": response.get("answer", "No answer found"),
            #     "sources": response.get("sources", []),
            #     "agent": self.name
            # }
        except Exception as e:
            self.logger.error(f"Document analysis failed: {e}")
            return {
                "answer": "Error analyzing documents",
                "error": str(e)
            }

    def _extract_sources(self, response) -> list:
        return [{
            "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
            "score": node.score,
            "metadata": node.metadata
        } for node in getattr(response, 'source_nodes', [])]