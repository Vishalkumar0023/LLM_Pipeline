#!/usr/bin/env python3
"""
Example: Run the LLM Pipeline on a PDF (or any file).

Usage:
    python3 example_llm_pipeline.py your_file.pdf
    python3 example_llm_pipeline.py paper.pdf notes.txt https://example.com
    python3 example_llm_pipeline.py ./docs_folder/
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import LLMPipeline


def main():
    # ─── Get sources from command line ───────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: python3 example_llm_pipeline.py <file.pdf> [more files...]")
        print("\nExamples:")
        print("  python3 example_llm_pipeline.py research_paper.pdf")
        print("  python3 example_llm_pipeline.py paper.pdf notes.md data.json")
        print(
            "  python3 example_llm_pipeline.py https://en.wikipedia.org/wiki/Machine_learning"
        )
        print("  python3 example_llm_pipeline.py ./my_docs/  (whole folder)")
        sys.exit(1)

    sources = sys.argv[1:]
    print(f"\n📥 Sources: {sources}\n")

    # ─── Initialize Pipeline ─────────────────────────────────────────
    llm = LLMPipeline(registry_dir="./llm_datasets", log_dir="./llm_logs")

    # ─── Option A: Full Pipeline (one call) ──────────────────────────
    result = llm.run_full_pipeline(
        sources=sources,
        version="v1.0.0",
        output_dir="./llm_output",
        model="meta-llama/Meta-Llama-3-8B",
        method="lora",  # or "qlora" for 4-bit
        domain="general",  # change to your domain: "medical", "legal", etc.
        template="alpaca",  # or "chatml", "sharegpt"
        chunk_method="sliding_window",
        chunk_size=512,
        min_quality_score=0.3,
        description="My first dataset",
    )

    # ─── Show outputs ────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("📂 OUTPUT FILES:")
    print("=" * 60)
    for key, path in result.items():
        size = os.path.getsize(path)
        print(f"  📄 {key}: {path} ({size:,} bytes)")

    print("\n🎯 Next steps:")
    print("  1. Review: ./llm_output/training_data.jsonl")
    print("  2. Review: ./llm_output/training_config.json")
    print("  3. To train (on GPU): python3 ./llm_output/train.py")
    print(
        "  4. Install GPU deps: pip install -r ./llm_output/requirements_training.txt"
    )


def step_by_step_example():
    """
    Alternative: Step-by-step control (uncomment main() call below to use).
    """
    from data_pipeline import (
        DocumentIngestor,
        TextChunker,
        InstructFormatter,
        QualityScorer,
        DatasetRegistry,
        FineTuneConfig,
        LLMMonitor,
    )

    # Layer 1: Ingest
    ingestor = DocumentIngestor()
    docs = ingestor.ingest(["your_file.pdf"])
    ingestor.print_summary()

    # Layer 2: Chunk
    chunker = TextChunker(method="sliding_window", chunk_size=512, overlap=64)
    chunks = chunker.chunk_documents(docs)
    chunker.print_summary()

    # Layer 2b: Format
    formatter = InstructFormatter(template="alpaca")
    pairs = formatter.format_chunks(chunks, domain="your domain")
    formatter.export_jsonl(pairs, "./output/training_data.jsonl")
    formatter.print_summary()

    # Layer 3: Quality
    scorer = QualityScorer(min_quality_score=0.4)
    scored = scorer.score(pairs)
    filtered = scorer.filter(scored)
    scorer.print_summary()

    # Layer 4: Version
    registry = DatasetRegistry("./llm_datasets")
    registry.register(filtered, version="v1.0.0", description="First pass")
    registry.print_summary()

    # Layer 5: Config
    config = FineTuneConfig(model_name="meta-llama/Meta-Llama-3-8B", method="lora")
    config.set_lora_params(r=16, alpha=32)
    config.export("./training_output", dataset_path="./output/training_data.jsonl")

    # Layer 6: Monitor (after you train)
    monitor = LLMMonitor("./llm_logs")
    monitor.log_run("v1.0", {"loss": 0.5, "perplexity": 12.0, "bleu": 0.3})
    trigger = monitor.check_retraining_triggers()
    print(f"\n🔄 Should retrain? {trigger['should_retrain']}")
    monitor.generate_report("./llm_logs/report.md")


if __name__ == "__main__":
    main()
    # step_by_step_example()  # Uncomment for step-by-step control
