import re
from typing import List, Dict, Any

class BenchmarkMetrics:
    """
    Computes rigorous retrieval, generation, and latency metrics for benchmark scenarios.
    """

    @staticmethod
    def calculate_hit_rate(retrieved_pages: List[int], expected_pages: List[int]) -> float:
        """
        Hit Rate @ k: checks if at least one expected page is present in retrieved chunks.
        """
        if not expected_pages:
            return 1.0  # Out of scope query
        intersection = set(retrieved_pages).intersection(set(expected_pages))
        return 1.0 if len(intersection) > 0 else 0.0

    @staticmethod
    def calculate_keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
        """
        Computes the ratio of expected semantic keywords found in the generated answer.
        """
        if not expected_keywords:
            return 1.0
        answer_lower = answer.lower()
        matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        return matched / len(expected_keywords)

    @staticmethod
    def evaluate_fallback(answer: str, is_out_of_scope: bool, fallback_triggered: bool) -> float:
        """
        Tests whether negative out-of-scope queries were safely rejected without hallucinations.
        """
        if not is_out_of_scope:
            return 1.0
        
        # Safe phrases for fallback
        safe_phrases = [
            "not found", "does not contain", "not mentioned", "no information",
            "not provided", "not available", "unable to find", "does not specify"
        ]
        answer_lower = answer.lower()
        text_fallback_detected = any(phrase in answer_lower for phrase in safe_phrases)
        
        return 1.0 if (fallback_triggered or text_fallback_detected) else 0.0

    @classmethod
    def evaluate_scenario(
        cls,
        scenario: Dict[str, Any],
        retrieved_pages: List[int],
        generated_answer: str,
        retrieval_latency_ms: float,
        fallback_triggered: bool
    ) -> Dict[str, Any]:
        """
        Evaluates a single test scenario and computes a composite scorecard.
        """
        is_out_of_scope = scenario.get("is_out_of_scope", False)
        expected_pages = scenario.get("expected_pages", [])
        expected_keywords = scenario.get("expected_answer_keywords", [])

        hit_rate = cls.calculate_hit_rate(retrieved_pages, expected_pages)
        keyword_cov = cls.calculate_keyword_coverage(generated_answer, expected_keywords)
        fallback_score = cls.evaluate_fallback(generated_answer, is_out_of_scope, fallback_triggered)
        latency_passed = retrieval_latency_ms < 200.0

        if is_out_of_scope:
            passed = fallback_score >= 1.0
        else:
            passed = (hit_rate >= 1.0 or keyword_cov >= 0.5) and (keyword_cov >= 0.4)

        return {
            "scenario_id": scenario["id"],
            "category": scenario["category"],
            "question": scenario["question"],
            "passed": passed,
            "hit_rate": hit_rate,
            "keyword_coverage": round(keyword_cov, 3),
            "fallback_score": fallback_score,
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "latency_under_200ms": latency_passed,
            "generated_answer": generated_answer,
            "retrieved_pages": retrieved_pages
        }
