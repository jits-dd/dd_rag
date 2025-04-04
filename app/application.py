from typing import Literal, Dict, Any, Optional
from llama_index.core.schema import NodeWithScore
from data_pipeline.evaluator import PipelineEvaluator
from typing import List
from config.settings import settings
import logging

class AdvancedRAGApplication:
    def __init__(
            self,
            query_engine,
            mode: Literal["query", "agent"] = "query",
            enable_evaluation: Optional[bool] = None
    ):
        self.mode = mode
        self.query_engine = query_engine
        self.enable_evaluation = enable_evaluation if enable_evaluation is not None else settings.ENABLE_EVALUATION
        self.logger = logging.getLogger(__name__)

        # print(mode)

        if self.enable_evaluation:
            self.evaluator = PipelineEvaluator()

        if mode == "agent":
            from orchestrator.agent import AdvancedConversationAgent
            self.orchestrator = AdvancedConversationAgent(query_engine)
        else:
            self.orchestrator = query_engine

    def query(self, query_str: str, evaluate: Optional[bool] = None) -> Dict[str, Any]:

        """Process query with optional evaluation"""
        evaluate = evaluate if evaluate is not None else self.enable_evaluation

        try:
            print("inside application query")
            response = self.orchestrator.query(query_str)
            print("after application query")

            result = {
                "answer": str(response),
                "sources": self._extract_sources(response),
                "metadata": getattr(response, "metadata", {})
            }

            if evaluate:
                eval_results, score = self.evaluator.evaluate_response(
                    query_str,
                    str(response),
                    [n.text for n in response.source_nodes]
                )
                result["evaluation"] = {
                    "results": eval_results,
                    "score": score
                }

            return result

        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return {
                "answer": "Sorry, I encountered an error processing your request.",
                "error": str(e)
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

            # Add conversation-specific metadata
            if node.metadata.get("is_conversation", False):
                source_info["type"] = "conversation"
                source_info["speakers"] = node.metadata.get("speakers", "unknown")
            else:
                source_info["type"] = "document"

            sources.append(source_info)

        return sources