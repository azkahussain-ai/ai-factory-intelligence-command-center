"""Stage IV tests - GenAI + RAG pipeline.

Run: pytest tests/test_stage4.py -v
(pytest CLI is unavailable in this sandbox, same as Stage II/III - these
functions take no fixtures so they can also be called directly; see
docs/HANDOFF.md for how they were verified.)
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunking import chunk_pages  # noqa: E402
from src.rag.document_loader import DocumentLoadError, load_knowledge_base_pdfs  # noqa: E402
from src.rag.generator import (  # noqa: E402
    NO_EVIDENCE_MESSAGE,
    generate_grounded_answer,
    generate_unsupported_answer,
)
from src.rag.rag_pipeline import RAGPipeline  # noqa: E402
from src.rag.retriever import RetrievedEvidence  # noqa: E402
from src.rag.vector_store import NumpyVectorStore  # noqa: E402


def test_knowledge_base_pdfs_exist_and_extract():
    kb_dir = PROJECT_ROOT / "knowledge_base"
    assert kb_dir.exists(), "run `python -m src.rag.build_knowledge_base_pdfs` first"
    pages = load_knowledge_base_pdfs(kb_dir)
    assert len(pages) > 0
    documents = {p.document for p in pages}
    assert documents == {
        "machine_operation_manual.pdf",
        "preventive_maintenance_sop.pdf",
        "safety_emergency_guidelines.pdf",
    }
    for p in pages:
        assert p.text.strip(), f"{p.document} page {p.page} extracted empty text"


def test_document_loader_raises_on_missing_directory(tmp_path=None):
    missing_dir = PROJECT_ROOT / "knowledge_base" / "__does_not_exist__"
    try:
        load_knowledge_base_pdfs(missing_dir)
        assert False, "expected DocumentLoadError"
    except DocumentLoadError:
        pass


def test_chunks_carry_traceable_metadata():
    pages = load_knowledge_base_pdfs()
    chunks = chunk_pages(pages)
    assert len(chunks) > 0
    for c in chunks:
        assert c.document
        assert c.page >= 1
        assert c.text.strip()
    # At least some chunks must have a detected section (the operation
    # manual and SOP both use "Section N: ..." headings).
    sectioned = [c for c in chunks if c.section]
    assert len(sectioned) > 0
    assert any("Section" in c.section for c in sectioned)


def test_vector_store_save_and_load_round_trip(tmp_path):
    pages = load_knowledge_base_pdfs()
    chunks = chunk_pages(pages)
    from src.rag.embeddings import TfidfEmbedder
    embedder = TfidfEmbedder()
    vectors = embedder.fit_transform([c.text for c in chunks])

    store = NumpyVectorStore()
    store.build(vectors, chunks)
    store.save(tmp_path)

    reloaded = NumpyVectorStore.load(tmp_path)
    assert len(reloaded) == len(store)

    query_vec = embedder.embed_query("vibration danger threshold")
    results_before = store.search(query_vec, top_k=3)
    results_after = reloaded.search(query_vec, top_k=3)
    assert [c.chunk_id for c, _ in results_before] == [c.chunk_id for c, _ in results_after]


def test_retrieval_finds_relevant_evidence_for_vibration_query():
    pipeline = RAGPipeline()
    result = pipeline.answer("What should the operator do if vibration is dangerously high?")
    assert result["grounded"] is True
    assert len(result["sources"]) > 0
    # The most relevant source should come from the SOP or the operation
    # manual, both of which discuss vibration thresholds.
    top_source = result["sources"][0]
    assert top_source["document"] in {
        "preventive_maintenance_sop.pdf",
        "machine_operation_manual.pdf",
    }
    assert "vibration" in top_source["evidence"].lower() or "DANGER" in top_source["evidence"]


def test_retrieval_returns_no_evidence_for_unrelated_query():
    pipeline = RAGPipeline()
    result = pipeline.answer("What is the recipe for chocolate cake?")
    assert result["grounded"] is False
    assert result["sources"] == []
    assert result["answer"] == NO_EVIDENCE_MESSAGE


def test_generator_grounded_answer_cites_evidence_text():
    evidence = [
        RetrievedEvidence(
            document="preventive_maintenance_sop.pdf",
            page=1,
            section="Section 3: Response to a DANGER-level Reading",
            evidence="reduce the machine's load or production rate immediately",
            score=0.5,
        )
    ]
    result = generate_grounded_answer("What should I do?", evidence)
    assert result["used_live_llm"] is False
    assert "reduce the machine's load" in result["answer"]
    assert "preventive_maintenance_sop.pdf" in result["answer"]


def test_generator_handles_missing_evidence():
    result = generate_grounded_answer("unanswerable question", [])
    assert result["answer"] == NO_EVIDENCE_MESSAGE


def test_unsupported_answer_has_no_document_citation_or_specific_threshold():
    """The no-RAG baseline must not be able to cite a specific factory
    document or the specific numeric thresholds that only exist in the
    knowledge base - that is the whole point of the comparison."""
    answer = generate_unsupported_answer(
        "What should the operator do if machine vibration becomes dangerously high?"
    )
    assert ".pdf" not in answer
    for forbidden in ["3.2 mm/s", "4.7 mm/s", "Section 3"]:
        assert forbidden not in answer


def test_rag_vs_no_rag_demo_report_is_grounded_vs_generic():
    from src.rag.demo_rag_vs_no_rag import run_demo
    result = run_demo()
    assert result["without_rag"]["sources"] == []
    assert len(result["with_rag"]["sources"]) > 0
    assert result["with_rag"]["grounded"] is True


def test_stage4_structured_output_schema():
    pipeline = RAGPipeline()
    result = pipeline.answer(
        "What should the operator do if machine vibration becomes dangerously high?",
        machine_id="M-01",
    )
    for key in ("question", "answer", "grounded", "used_live_llm", "sources"):
        assert key in result
    for source in result["sources"]:
        for key in ("document", "page", "section", "evidence", "score"):
            assert key in source
    assert result["machine_context"] is not None
    assert "M-01" in result["machine_context"]


def test_stage1_to_3_outputs_unchanged_by_stage4():
    """Recompute md5 checksums of every Stage I-III file and compare against
    the baseline recorded before Stage IV work began."""
    baseline_path = PROJECT_ROOT.parent / "stage1to3_checksums_before.md5"
    if not baseline_path.exists():
        # Baseline file lives outside the repo (dev-sandbox artifact); skip
        # gracefully if it isn't present (e.g. a fresh checkout).
        return
    lines = baseline_path.read_text().strip().split("\n")
    mismatches = []
    for line in lines:
        expected_hash, rel_path = line.split(None, 1)
        rel_path = rel_path.strip()
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            mismatches.append(f"MISSING: {rel_path}")
            continue
        actual_hash = hashlib.md5(full_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            mismatches.append(f"CHANGED: {rel_path}")
    assert not mismatches, f"Stage I-III files modified by Stage IV: {mismatches}"
