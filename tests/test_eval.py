"""Tests for the evaluation metrics and prompt/groundedness helpers."""

from eval.generation_eval import lexical_groundedness
from eval.retrieval_eval import precision_at_k, recall_at_k
from src.generate import build_prompt, format_context


class TestRetrievalMetrics:
    def test_precision_at_k(self):
        # 2 of the top 3 are relevant.
        assert precision_at_k(["a", "b", "c"], ["a", "c", "z"], k=3) == 2 / 3

    def test_precision_at_k_perfect(self):
        assert precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_precision_at_k_zero_k(self):
        assert precision_at_k(["a"], ["a"], k=0) == 0.0

    def test_recall_at_k(self):
        # Found 1 of 2 relevant docs in the top 3.
        assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5

    def test_recall_at_k_no_relevant(self):
        assert recall_at_k(["a", "b"], [], k=2) == 0.0


class TestPromptBuilding:
    def test_context_is_numbered(self):
        docs = [{"text": "alpha"}, {"text": "beta"}]
        ctx = format_context(docs)
        assert "[1] alpha" in ctx
        assert "[2] beta" in ctx

    def test_prompt_contains_question_and_context(self):
        prompt = build_prompt("what skills?", [{"text": "python"}])
        assert "what skills?" in prompt
        assert "python" in prompt
        assert "citations" in prompt.lower()


class TestGroundedness:
    def test_fully_grounded(self):
        docs = [{"text": "python and sql are required"}]
        assert lexical_groundedness("python sql", docs) == 1.0

    def test_partially_grounded(self):
        docs = [{"text": "python required"}]
        # "python" grounded, "rust" not → 1/2.
        assert lexical_groundedness("python rust", docs) == 0.5

    def test_empty_answer(self):
        assert lexical_groundedness("", [{"text": "anything"}]) == 0.0
