import json
from dataclasses import dataclass
from typing import Dict


@dataclass
class EvaluationResult:
    response_quality: float
    factual_consistency: float
    hallucination_risk: float
    safety_alignment: float

    def overall_score(self) -> float:
        return round(
            (
                self.response_quality +
                self.factual_consistency +
                self.safety_alignment -
                self.hallucination_risk
            ) / 3,
            2
        )

    def to_dict(self) -> Dict:
        return {
            "response_quality": self.response_quality,
            "factual_consistency": self.factual_consistency,
            "hallucination_risk": self.hallucination_risk,
            "safety_alignment": self.safety_alignment,
            "overall_score": self.overall_score()
        }


class ResponseEvaluator:

    def evaluate(self, prompt: str, response: str) -> EvaluationResult:

        response_quality = self.score_response_quality(response)
        factual_consistency = self.score_factual_consistency(response)
        hallucination_risk = self.score_hallucination_risk(response)
        safety_alignment = self.score_safety_alignment(response)

        return EvaluationResult(
            response_quality=response_quality,
            factual_consistency=factual_consistency,
            hallucination_risk=hallucination_risk,
            safety_alignment=safety_alignment
        )

    def score_response_quality(self, response: str) -> float:
        return min(len(response) / 100, 10)

    def score_factual_consistency(self, response: str) -> float:
        return 8.5

    def score_hallucination_risk(self, response: str) -> float:
        return 2.0

    def score_safety_alignment(self, response: str) -> float:
        return 9.0


if __name__ == "__main__":

    prompt = "Explain reinforcement learning from human feedback."

    response = (
        "Reinforcement Learning from Human Feedback (RLHF) "
        "is a machine learning alignment technique used to "
        "optimize language models using human preference data."
    )

    evaluator = ResponseEvaluator()

    result = evaluator.evaluate(prompt, response)

    print(json.dumps(result.to_dict(), indent=4))
