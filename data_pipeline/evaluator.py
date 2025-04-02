from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner
)
from typing import Tuple, Dict, Any, List

class PipelineEvaluator:
    def __init__(self):
        self.faithfulness_eval = FaithfulnessEvaluator()
        self.relevancy_eval = RelevancyEvaluator()
        self.eval_runner = BatchEvalRunner(
            {"faithfulness": self.faithfulness_eval, "relevancy": self.relevancy_eval},
            workers=2
        )

    def evaluate_response(
            self,
            query: str,
            response: str,
            contexts: List[str]
    ) -> Tuple[Dict[str, Any], float]:
        """Evaluate response quality"""
        eval_results = self.eval_runner.evaluate(
            queries=[query],
            responses=[response],
            contexts=[contexts]
        )

        # Calculate overall score
        score = (
                        eval_results["faithfulness"].passing
                        + eval_results["relevancy"].passing
                ) / 2

        return eval_results, score