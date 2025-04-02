from typing import Literal, Dict, Any
from llama_index.core.schema import NodeWithScore
from data_pipeline.evaluator import PipelineEvaluator
from typing import List

class AdvancedRAGApplication:
    def __init__(self, query_engine, mode: Literal["query", "agent"] = "query"):
        self.mode = mode
        self.query_engine = query_engine
        self.evaluator = PipelineEvaluator()

        if mode == "agent":
            from orchestrator.agent import AdvancedRAGAgent
            self.orchestrator = AdvancedRAGAgent(query_engine)
        else:
            self.orchestrator = query_engine

    def query(self, query_str: str, evaluate: bool = False) -> Dict[str, Any]:
        """Process query with optional evaluation"""
        response = self.orchestrator.query(query_str)

        result = {
            "answer": str(response),
            "sources": self._extract_sources(response)
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

    def _extract_sources(self, response) -> List[Dict]:
        return [
            {
                "text": node.text,
                "metadata": node.metadata,
                "score": node.score if hasattr(node, "score") else None
            }
            for node in getattr(response, "source_nodes", [])
        ]