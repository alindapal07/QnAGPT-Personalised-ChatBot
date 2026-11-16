"""Comprehensive test suite for Corrective Retrieval-Augmented Generation (CRAG)."""

from typing import Any, Dict, List, Optional
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from qna_bot.agent.graph import build_agent_graph
from qna_bot.agent.nodes import AgentNodeSuite
from qna_bot.agent.routing import route_after_grading
from qna_bot.agent.state import AgentState
from qna_bot.config.settings import AppSettings
from qna_bot.rag.crag.document_filter import DocumentFilter
from qna_bot.rag.crag.evaluator import DocumentEvaluator
from qna_bot.rag.crag.evidence import EvidenceAssembler
from qna_bot.rag.crag.metrics import CRAGMetricsTracker
from qna_bot.rag.crag.models import (
    CRAGDecision,
    CRAGDocument,
    CRAGEvaluation,
    EvidenceBundle,
    EvidenceItem,
    EvidenceStatus,
    RelevanceGrade,
    RewrittenQuery,
    _clamp_unit_interval,
)
from qna_bot.rag.crag.pipeline import CRAGPipeline
from qna_bot.rag.crag.query_rewriter import QueryRewriter
from qna_bot.rag.crag.routing import CRAGRouter
from qna_bot.rag.vector_store import RetrievedChunk


class TestCRAGModels(unittest.TestCase):
    """Test data model validation, bounds clamping, and serialization."""

    def test_score_clamping(self) -> None:
        self.assertEqual(_clamp_unit_interval(-0.5), 0.0)
        self.assertEqual(_clamp_unit_interval(1.5), 1.0)
        self.assertEqual(_clamp_unit_interval(0.75), 0.75)
        self.assertEqual(_clamp_unit_interval("invalid"), 0.0)

        doc = CRAGDocument(content="Text", score=1.8)
        self.assertEqual(doc.score, 1.0)

        doc_neg = CRAGDocument(content="Text", score=-0.2)
        self.assertEqual(doc_neg.score, 0.0)

    def test_crag_evaluation_model_and_serialization(self) -> None:
        eval_obj = CRAGEvaluation(
            relevance_grade=RelevanceGrade.CORRECT,
            relevance_score=0.92,
            confidence_score=0.88,
            query_coverage=0.95,
            factual_sufficiency=0.85,
            contradiction_detected=False,
            relevant_indices=[0, 1],
            salient_snippets=["Important fact here."],
            missing_aspects=[],
            explanation="High relevance and coverage.",
            used_llm=True,
        )
        data = eval_obj.to_dict()
        self.assertEqual(data["relevance_grade"], "CORRECT")
        self.assertEqual(data["relevance_score"], 0.92)
        self.assertEqual(data["relevant_indices"], [0, 1])

        # Roundtrip deserialization
        restored = CRAGEvaluation.model_validate(data)
        self.assertEqual(restored.relevance_grade, RelevanceGrade.CORRECT)
        self.assertEqual(restored.salient_snippets, ["Important fact here."])

    def test_evidence_bundle_and_items(self) -> None:
        item = EvidenceItem(
            text="Revenue increased 20% in Q3.",
            source="annual_report.pdf",
            page=4,
            score=0.89,
            metadata={"year": 2024},
        )
        bundle = EvidenceBundle(
            status=EvidenceStatus.VALID,
            items=[item],
            synthesized_context="[1] Source: annual_report.pdf (Page 4)\nRevenue increased 20% in Q3.",
            overall_confidence=0.89,
            query_coverage=1.0,
            sources=[{"source": "annual_report.pdf", "page": 4, "score": 0.89}],
        )
        b_dict = bundle.to_dict()
        self.assertEqual(b_dict["status"], "VALID")
        self.assertEqual(len(b_dict["items"]), 1)
        self.assertEqual(b_dict["items"][0]["source"], "annual_report.pdf")


class TestDocumentEvaluator(unittest.TestCase):
    """Test multi-dimensional evaluation: CORRECT, AMBIGUOUS, INCORRECT, CONTRADICTORY."""

    def setUp(self) -> None:
        self.evaluator = DocumentEvaluator(
            api_key="",
            relevance_threshold=0.70,
            confidence_threshold=0.70,
            coverage_threshold=0.65,
        )

    def test_heuristic_evaluate_correct(self) -> None:
        query = "What is the capital and population of France?"
        docs = [
            CRAGDocument(
                content="The capital of France is Paris. Paris has an estimated metropolitan population of over 12 million.",
                source="france_facts.txt",
                score=0.88,
            )
        ]
        result = self.evaluator.evaluate_batch(query, docs)
        self.assertEqual(result.relevance_grade, RelevanceGrade.CORRECT)
        self.assertGreaterEqual(result.relevance_score, 0.70)
        self.assertGreaterEqual(result.query_coverage, 0.65)
        self.assertIn(0, result.relevant_indices)
        self.assertFalse(result.contradiction_detected)

    def test_heuristic_evaluate_ambiguous(self) -> None:
        query = "What are the revenue figures and cloud gross margins for Q3?"
        docs = [
            CRAGDocument(
                content="The company posted strong financial performance overall in Q3.",
                source="finance.txt",
                score=0.55,
            )
        ]
        result = self.evaluator.evaluate_batch(query, docs)
        self.assertEqual(result.relevance_grade, RelevanceGrade.AMBIGUOUS)
        self.assertTrue(len(result.missing_aspects) > 0)

    def test_heuristic_evaluate_incorrect(self) -> None:
        query = "What is the atomic mass of Gold?"
        docs = [
            CRAGDocument(
                content="Baking a chocolate cake requires flour, cocoa powder, sugar, and eggs.",
                source="recipes.txt",
                score=0.10,
            )
        ]
        result = self.evaluator.evaluate_batch(query, docs)
        self.assertEqual(result.relevance_grade, RelevanceGrade.INCORRECT)
        self.assertEqual(result.relevant_indices, [])

    def test_heuristic_evaluate_contradiction(self) -> None:
        query = "Is TLS 1.3 encryption allowed on the internal network?"
        docs = [
            CRAGDocument(
                content="TLS 1.3 protocol is confirmed and allowed for all internal cluster nodes.",
                source="sec_policy_a.txt",
                score=0.85,
            ),
            CRAGDocument(
                content="TLS 1.3 protocol is denied and forbidden due to inspection firewall incompatibility.",
                source="sec_policy_b.txt",
                score=0.82,
            ),
        ]
        result = self.evaluator.evaluate_batch(query, docs)
        self.assertEqual(result.relevance_grade, RelevanceGrade.CONTRADICTORY)
        self.assertTrue(result.contradiction_detected)
        self.assertTrue(len(result.contradiction_details) > 0)

    def test_empty_documents_returns_incorrect(self) -> None:
        result = self.evaluator.evaluate_batch("Any query", [])
        self.assertEqual(result.relevance_grade, RelevanceGrade.INCORRECT)
        self.assertEqual(result.relevance_score, 0.0)

    def test_llm_evaluate_structured_output_mock(self) -> None:
        evaluator = DocumentEvaluator(
            api_key="gsk_test",
            relevance_threshold=0.70,
            confidence_threshold=0.70,
            coverage_threshold=0.65,
        )
        mock_output = {
            "relevance_score": 0.95,
            "confidence_score": 0.90,
            "query_coverage": 0.92,
            "factual_sufficiency": 0.88,
            "contradiction_detected": False,
            "temporal_validity": None,
            "relevant_indices": [0],
            "salient_snippets": ["Salient fact"],
            "missing_aspects": [],
            "contradiction_details": [],
            "grade": "CORRECT",
            "explanation": "Grounded match",
        }

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = mock_output
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        with patch.object(evaluator, "_make_llm", return_value=mock_llm):
            docs = [CRAGDocument(content="Fact text", source="doc.txt", score=0.9)]
            res = evaluator.evaluate_batch("test query", docs)
            self.assertEqual(res.relevance_grade, RelevanceGrade.CORRECT)
            self.assertEqual(res.relevance_score, 0.95)
            self.assertTrue(res.used_llm)


class TestQueryRewriter(unittest.TestCase):
    """Test conversational resolution, decomposition, and drift prevention."""

    def setUp(self) -> None:
        self.rewriter = QueryRewriter(api_key="")

    def test_heuristic_strips_conversational_filler(self) -> None:
        raw_query = "Can you please tell me what is the latest pricing plan?"
        res = self.rewriter.rewrite(raw_query)
        self.assertNotIn("can you please tell me", res.rewritten_query.lower())
        self.assertIn("pricing", res.rewritten_query.lower())

    def test_heuristic_decomposes_compound_query(self) -> None:
        query = "Explain Kubernetes pod architecture and explain Docker container security"
        res = self.rewriter.rewrite(query)
        self.assertGreater(len(res.subqueries), 0)
        self.assertIn("pod", res.extracted_keywords)
        self.assertIn("docker", res.extracted_keywords)

    def test_retry_incorporates_missing_aspects(self) -> None:
        query = "Company financials"
        evaluation = CRAGEvaluation(
            relevance_grade=RelevanceGrade.AMBIGUOUS,
            relevance_score=0.5,
            confidence_score=0.5,
            query_coverage=0.4,
            factual_sufficiency=0.3,
            missing_aspects=["net income", "operating margin"],
        )
        res = self.rewriter.rewrite(
            query=query,
            attempt=2,
            evaluation=evaluation,
            previous_rewrites=["Company financials overview"],
        )
        self.assertIn("net income", res.rewritten_query)


class TestDocumentFilterAndDeduplication(unittest.TestCase):
    """Test filtering noise and deduplicating chunks."""

    def setUp(self) -> None:
        self.filter = DocumentFilter()

    def test_filter_and_extract_salient_snippets(self) -> None:
        query = "What is the return policy duration?"
        docs = [
            CRAGDocument(
                content="Welcome to our store. We sell shoes. All customers may return items within 30 days of purchase. Thanks for visiting.",
                source="policy.txt",
                score=0.85,
            )
        ]
        snippets = self.filter.filter_and_extract_snippets(query, docs)
        self.assertTrue(any("30 days" in s for s in snippets))

    def test_deduplicate_identical_documents(self) -> None:
        docs = [
            CRAGDocument(content="Identical paragraph text repeated across multiple pages of report.", score=0.8),
            CRAGDocument(content="Identical paragraph text repeated across multiple pages of report.", score=0.8),
            CRAGDocument(content="Unique paragraph text covering separate quarterly statistics.", score=0.85),
        ]
        deduped = self.filter.deduplicate_documents(docs)
        self.assertEqual(len(deduped), 2)


class TestEvidenceAssembler(unittest.TestCase):
    """Test assembling evidence bundles and citations."""

    def setUp(self) -> None:
        self.assembler = EvidenceAssembler()

    def test_assemble_from_correct_evaluation(self) -> None:
        evaluation = CRAGEvaluation(
            relevance_grade=RelevanceGrade.CORRECT,
            relevance_score=0.90,
            confidence_score=0.85,
            query_coverage=0.95,
            factual_sufficiency=0.90,
            relevant_indices=[0],
        )
        docs = [
            CRAGDocument(
                content="Python 3.12 was released in October 2023.",
                source="python_release.md",
                page=1,
                score=0.92,
            )
        ]
        bundle = self.assembler.assemble_from_evaluation(evaluation, docs)
        self.assertEqual(bundle.status, EvidenceStatus.VALID)
        self.assertFalse(bundle.requires_web_search)
        self.assertFalse(bundle.requires_rewrite)
        self.assertEqual(len(bundle.sources), 1)
        self.assertEqual(bundle.sources[0]["source"], "python_release.md")

    def test_assemble_from_web_results(self) -> None:
        web_hits = [
            {"title": "Python 3.12 Release Notes", "url": "https://python.org/3.12", "snippet": "New features in 3.12 include perf profiler."}
        ]
        opened = [
            {"title": "Python Official", "url": "https://python.org/3.12", "content": "Full release announcement for Python 3.12...", "success": True}
        ]
        bundle = self.assembler.assemble_from_web_results(web_hits, opened)
        self.assertEqual(bundle.status, EvidenceStatus.VALID)
        self.assertEqual(len(bundle.items), 1)
        self.assertEqual(bundle.sources[0]["type"], "web")


class TestCRAGRouter(unittest.TestCase):
    """Test bounded routing transitions: CORRECT, AMBIGUOUS, INCORRECT, CONTRADICTORY."""

    def setUp(self) -> None:
        self.router = CRAGRouter(
            max_retrieval_attempts=2,
            max_web_search_attempts=1,
            web_fallback_enabled=True,
        )

    def test_route_correct_returns_correct(self) -> None:
        eval_obj = CRAGEvaluation(relevance_grade=RelevanceGrade.CORRECT)
        decision = self.router.decide(eval_obj, retrieval_attempts=1)
        self.assertEqual(decision, CRAGDecision.CORRECT)

    def test_route_ambiguous_first_attempt_returns_ambiguous(self) -> None:
        eval_obj = CRAGEvaluation(relevance_grade=RelevanceGrade.AMBIGUOUS)
        decision = self.router.decide(eval_obj, retrieval_attempts=1)
        self.assertEqual(decision, CRAGDecision.AMBIGUOUS)

    def test_route_ambiguous_second_attempt_falls_back_to_web(self) -> None:
        eval_obj = CRAGEvaluation(relevance_grade=RelevanceGrade.AMBIGUOUS)
        decision = self.router.decide(eval_obj, retrieval_attempts=2, web_search_attempts=0, chat_mode="Auto (Agentic)")
        self.assertEqual(decision, CRAGDecision.INCORRECT)

    def test_route_incorrect_triggers_web_fallback(self) -> None:
        eval_obj = CRAGEvaluation(relevance_grade=RelevanceGrade.INCORRECT)
        decision = self.router.decide(eval_obj, retrieval_attempts=1, web_search_attempts=0, chat_mode="Auto (Agentic)")
        self.assertEqual(decision, CRAGDecision.INCORRECT)

    def test_route_contradiction_triggers_web_verification(self) -> None:
        eval_obj = CRAGEvaluation(
            relevance_grade=RelevanceGrade.CONTRADICTORY,
            contradiction_detected=True,
        )
        decision = self.router.decide(eval_obj, web_search_attempts=0, chat_mode="Auto (Agentic)")
        self.assertEqual(decision, CRAGDecision.CONTRADICTORY)

    def test_route_after_grading_integration(self) -> None:
        # Test routing function hooked to LangGraph
        state: AgentState = {
            "crag_decision": "AMBIGUOUS",
            "retry_count": 0,
            "chat_mode": "Auto (Agentic)",
        }
        dest = route_after_grading(state, max_retrieval_retries=2)
        self.assertEqual(dest, "rewrite_query")

        state["retry_count"] = 2
        dest_exhausted = route_after_grading(state, max_retrieval_retries=2)
        self.assertEqual(dest_exhausted, "web_search")


class TestCRAGMetricsTracker(unittest.TestCase):
    """Test telemetry recorder."""

    def test_metrics_recording_and_summary(self) -> None:
        tracker = CRAGMetricsTracker()
        tracker.record_evaluation(CRAGEvaluation(relevance_grade=RelevanceGrade.CORRECT), latency_ms=120.0)
        tracker.record_evaluation(CRAGEvaluation(relevance_grade=RelevanceGrade.AMBIGUOUS), latency_ms=180.0)
        tracker.record_rewrite()
        tracker.record_web_fallback()
        tracker.record_abstention()

        summary = tracker.get_summary()
        self.assertEqual(summary["total_evaluations"], 2)
        self.assertEqual(summary["grade_distribution"]["CORRECT"], 1)
        self.assertEqual(summary["grade_distribution"]["AMBIGUOUS"], 1)
        self.assertEqual(summary["query_rewrites"], 1)
        self.assertEqual(summary["web_fallbacks"], 1)
        self.assertEqual(summary["abstentions"], 1)
        self.assertAlmostEqual(summary["avg_evaluation_latency_ms"], 150.0)


class TestCRAGLangGraphIntegration(unittest.TestCase):
    """Test LangGraph node suite with CRAG and multi-tenant isolation."""

    def test_retrieve_documents_enforces_user_isolation(self) -> None:
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RetrievedChunk(
                content="Tenant A confidential report",
                metadata={"user_id": "tenant_a"},
                score=0.91,
                document_id="doc_1",
                source="report.pdf",
            )
        ]
        suite = AgentNodeSuite(
            retriever=mock_retriever,
            grader=MagicMock(),
            web_searcher=MagicMock(),
            tools=[],
            settings=AppSettings(groq_api_key="test", crag_enabled=True),
        )

        state: AgentState = {
            "user_query": "Summarize confidential report",
            "user_id": "tenant_a",
            "retrieval_attempt_count": 0,
        }
        out = suite.retrieve_documents(state)
        mock_retriever.retrieve.assert_called_once_with(
            query="Summarize confidential report",
            user_id="tenant_a",
            top_k=4,
        )
        self.assertEqual(out["retrieval_attempt_count"], 1)
        self.assertEqual(len(out["retrieved_documents"]), 1)

    def test_grade_documents_with_crag_produces_evaluation_and_bundle(self) -> None:
        mock_pipeline = MagicMock()
        mock_pipeline.evaluate_and_assemble.return_value = (
            CRAGEvaluation(
                relevance_grade=RelevanceGrade.CORRECT,
                relevance_score=0.95,
                confidence_score=0.90,
                query_coverage=1.0,
                factual_sufficiency=0.90,
                relevant_indices=[0],
            ),
            EvidenceBundle(
                status=EvidenceStatus.VALID,
                items=[EvidenceItem(text="Fact", source="doc.txt", score=0.95)],
                synthesized_context="[1] Source: doc.txt\nFact",
                sources=[{"source": "doc.txt", "score": 0.95}],
            ),
            CRAGDecision.CORRECT,
        )

        suite = AgentNodeSuite(
            retriever=MagicMock(),
            grader=MagicMock(),
            web_searcher=MagicMock(),
            tools=[],
            crag_pipeline=mock_pipeline,
            settings=AppSettings(groq_api_key="test", crag_enabled=True),
        )

        state: AgentState = {
            "user_query": "What is the fact?",
            "retrieved_documents": [{"content": "Fact", "source": "doc.txt", "score": 0.95}],
            "retrieval_attempt_count": 1,
        }
        res = suite.grade_documents(state)
        self.assertEqual(res["crag_decision"], "CORRECT")
        self.assertEqual(res["crag_status"], "VALID")
        self.assertEqual(len(res["relevant_documents"]), 1)
        self.assertEqual(len(res["document_evaluations"]), 1)

    def test_generate_answer_abstains_honestly_when_evidence_empty(self) -> None:
        suite = AgentNodeSuite(
            retriever=MagicMock(),
            grader=MagicMock(),
            web_searcher=MagicMock(),
            tools=[],
            settings=AppSettings(groq_api_key="test", crag_enabled=True, crag_abstention_enabled=True),
        )
        state: AgentState = {
            "user_query": "Who won the 2099 solar race?",
            "relevant_documents": [],
            "web_results": [],
            "opened_webpages": [],
            "verified_research": {},
            "chat_mode": "Auto (Agentic)",
        }
        res = suite.generate_answer(state)
        self.assertIn("could not find sufficient verifiable information", res["answer"].lower())
        self.assertTrue(res["validation_passed"])

    def test_pipeline_e2e_evaluation_and_metrics(self) -> None:
        evaluator = DocumentEvaluator(api_key="", relevance_threshold=0.70)
        pipeline = CRAGPipeline(evaluator=evaluator)
        pipeline.metrics.reset()

        query = "Explain solar panel efficiency rates"
        docs = [
            CRAGDocument(
                content="Standard commercial solar panel efficiency rates range from 17% to 22%.",
                source="solar_guide.txt",
                score=0.90,
            )
        ]
        eval_res, bundle, decision = pipeline.evaluate_and_assemble(query, docs)
        self.assertEqual(decision, CRAGDecision.CORRECT)
        self.assertEqual(bundle.status, EvidenceStatus.VALID)

        summary = pipeline.metrics.get_summary()
        self.assertEqual(summary["total_evaluations"], 1)
        self.assertEqual(summary["grade_distribution"]["CORRECT"], 1)


if __name__ == "__main__":
    unittest.main()
