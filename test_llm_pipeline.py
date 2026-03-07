#!/usr/bin/env python3
"""
Integration test for the LLM Pipeline.
Tests all 6 layers end-to-end with sample data.
"""

import os
import sys
import json
import shutil
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_DIR = tempfile.mkdtemp(prefix='llm_pipeline_test_')
PASS = "✅"
FAIL = "❌"
results = []


def report(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, passed))
    msg = f"  {status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


def setup_test_files():
    """Create sample test files."""
    # Plain text file
    txt_path = os.path.join(TEST_DIR, 'sample.txt')
    with open(txt_path, 'w') as f:
        f.write("""Machine Learning Overview

Machine learning is a subset of artificial intelligence that focuses on building systems 
that learn from data. These systems improve their performance over time without being 
explicitly programmed for every task.

Supervised learning is one of the most common approaches. In supervised learning, the model 
is trained on labeled data. The algorithm learns a mapping from inputs to outputs by 
examining many examples. Common algorithms include linear regression, decision trees, 
and neural networks.

Unsupervised learning works with unlabeled data. The goal is to discover hidden patterns 
or structures in the data. Clustering and dimensionality reduction are popular techniques.
K-means clustering groups similar data points together, while PCA reduces the number of 
features while preserving variance.

Deep learning uses neural networks with many layers. These deep neural networks have 
revolutionized fields like computer vision and natural language processing. Convolutional 
Neural Networks excel at image recognition tasks, while Recurrent Neural Networks and 
Transformers are designed for sequential data processing.

Transfer learning allows models trained on large datasets to be fine-tuned for specific 
tasks with limited data. This approach has become fundamental in modern NLP, where 
large language models like GPT and BERT are pre-trained on massive text corpora and 
then adapted for downstream tasks.
""")

    # Markdown file
    md_path = os.path.join(TEST_DIR, 'notes.md')
    with open(md_path, 'w') as f:
        f.write("""# Fine-Tuning Guide

## What is Fine-Tuning?

Fine-tuning is the process of taking a pre-trained model and adapting it to a specific 
task or domain. This is more efficient than training from scratch because the model 
already understands general language patterns.

## LoRA Method

Low-Rank Adaptation (LoRA) is an efficient fine-tuning technique. Instead of updating 
all model parameters, LoRA adds small trainable matrices to specific layers. This 
dramatically reduces memory requirements and training time.

The key parameters in LoRA are:
- Rank (r): Controls the dimensionality of the low-rank matrices
- Alpha: A scaling factor that balances the original and adapted weights
- Target modules: Which layers to apply LoRA to (typically attention layers)

## Best Practices

Start with a small learning rate (2e-4 or lower). Use gradient accumulation if your 
batch size is limited by GPU memory. Monitor validation loss to prevent overfitting.
Always evaluate on a held-out test set that represents your target distribution.
""")

    # JSON file
    json_path = os.path.join(TEST_DIR, 'data.json')
    with open(json_path, 'w') as f:
        json.dump([
            {"topic": "Gradient Descent", "content": "Gradient descent is an optimization algorithm used to minimize the loss function. It iteratively adjusts parameters in the direction of steepest descent. The learning rate controls the step size."},
            {"topic": "Backpropagation", "content": "Backpropagation is the algorithm used to compute gradients in neural networks. It applies the chain rule to calculate how each weight contributes to the overall error."}
        ], f, indent=2)

    return [txt_path, md_path, json_path]


def test_layer1_ingestion(sources):
    """Test Layer 1: Document Ingestion."""
    print("\n── Layer 1: Document Ingestion ──")
    from data_pipeline.document_ingestor import DocumentIngestor

    ingestor = DocumentIngestor()
    docs = ingestor.ingest(sources)

    report("Ingestor returns documents", len(docs) > 0, f"{len(docs)} docs")
    report("Each doc has 'text' field", all('text' in d for d in docs))
    report("Each doc has 'doc_id' field", all('doc_id' in d for d in docs))
    report("Each doc has 'source_type'", all('source_type' in d for d in docs))

    types = set(d['source_type'] for d in docs)
    report("Multiple source types detected", len(types) >= 2, f"types: {types}")

    stats = ingestor.get_stats()
    report("Stats track totals", stats['total_documents'] == len(docs))

    return docs


def test_layer2_chunking(docs):
    """Test Layer 2: Text Chunking."""
    print("\n── Layer 2: Text Chunking ──")
    from data_pipeline.text_chunker import TextChunker

    # Test sliding window
    chunker = TextChunker(method='sliding_window', chunk_size=300, overlap=50)
    chunks = chunker.chunk_documents(docs)

    report("Chunker returns chunks", len(chunks) > 0, f"{len(chunks)} chunks")
    report("Chunks have 'text' field", all('text' in c for c in chunks))
    report("Chunks have 'estimated_tokens'", all('estimated_tokens' in c for c in chunks))
    report("Chunks preserve source", all('source' in c for c in chunks))
    report("More chunks than docs", len(chunks) >= len(docs))

    stats = chunker.get_stats()
    report("Stats track chunk count", stats['total_chunks'] == len(chunks))

    # Test paragraph method
    chunker2 = TextChunker(method='paragraph', chunk_size=500)
    chunks2 = chunker2.chunk_documents(docs)
    report("Paragraph method works", len(chunks2) > 0, f"{len(chunks2)} chunks")

    return chunks


def test_layer3_formatting(chunks):
    """Test Layer 3: Instruction Formatting."""
    print("\n── Layer 3: Instruction Formatting ──")
    from data_pipeline.instruct_formatter import InstructFormatter

    # Test Alpaca format
    formatter = InstructFormatter(template='alpaca')
    pairs = formatter.format_chunks(chunks, domain='machine learning')

    report("Formatter returns pairs", len(pairs) > 0, f"{len(pairs)} pairs")
    report("Alpaca has 'instruction'", all('instruction' in p for p in pairs))
    report("Alpaca has 'output'", all('output' in p for p in pairs))

    # Test JSONL export
    jsonl_path = os.path.join(TEST_DIR, 'test_export.jsonl')
    formatter.export_jsonl(pairs, jsonl_path)
    report("JSONL export created", os.path.exists(jsonl_path))

    # Validate JSONL
    with open(jsonl_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    report("JSONL is valid JSON", len(lines) == len(pairs))

    # Test ChatML format
    formatter2 = InstructFormatter(template='chatml')
    pairs2 = formatter2.format_chunks(chunks[:2], domain='AI')
    report("ChatML has 'messages'", all('messages' in p for p in pairs2))

    stats = formatter.get_stats()
    report("Stats track pair count", stats['total_pairs'] == len(pairs))

    return pairs


def test_layer4_quality(pairs):
    """Test Layer 4: Quality Scoring."""
    print("\n── Layer 4: Quality Scoring ──")
    from data_pipeline.quality_scorer import QualityScorer

    scorer = QualityScorer(min_quality_score=0.3)
    scored = scorer.score(pairs)

    report("Scorer returns scored pairs", len(scored) == len(pairs))
    report("Pairs have 'quality' dict", all('quality' in p for p in scored))
    report("Quality has 'overall_score'",
           all('overall_score' in p['quality'] for p in scored))

    # Check scores are in valid range
    scores = [p['quality']['overall_score'] for p in scored]
    report("Scores in [0, 1]", all(0 <= s <= 1 for s in scores))

    # Filter
    filtered = scorer.filter(scored, min_score=0.3)
    report("Filter removes low quality", len(filtered) <= len(scored),
           f"{len(filtered)}/{len(scored)} kept")

    stats = scorer.get_stats()
    report("Stats track totals", stats['total_samples'] == len(pairs))

    return filtered


def test_layer5_versioning(data):
    """Test Layer 5: Dataset Versioning."""
    print("\n── Layer 5: Dataset Versioning ──")
    from data_pipeline.dataset_registry import DatasetRegistry

    reg_dir = os.path.join(TEST_DIR, 'registry')
    registry = DatasetRegistry(reg_dir)

    # Register v1
    meta1 = registry.register(
        data, version="v1.0.0",
        description="Initial test dataset",
        tags=["test"]
    )
    report("V1 registered", meta1['version'] == 'v1.0.0')
    report("V1 has hash", len(meta1['data_hash']) == 64)
    report("V1 row count matches", meta1['row_count'] == len(data))

    # Register v2 (subset)
    meta2 = registry.register(
        data[:max(1, len(data) // 2)],
        version="v1.1.0",
        description="Filtered subset",
        parent_version="v1.0.0"
    )
    report("V2 registered", meta2['version'] == 'v1.1.0')

    # List versions
    versions = registry.list_versions()
    report("List shows 2 versions", len(versions) == 2)

    # Load version
    loaded = registry.load_version("v1.0.0")
    report("Load returns data", len(loaded) == len(data))

    # Diff
    diff = registry.diff("v1.0.0", "v1.1.0")
    report("Diff has counts", 'added' in diff and 'removed' in diff)

    # Rollback
    registry.rollback("v1.0.0")
    report("Rollback sets latest", registry._registry['latest'] == 'v1.0.0')

    return registry


def test_layer6_finetune_config():
    """Test Layer 6a: Fine-Tune Config."""
    print("\n── Layer 6a: Fine-Tune Config ──")
    from data_pipeline.finetune_config import FineTuneConfig

    config = FineTuneConfig(
        model_name='meta-llama/Meta-Llama-3-8B',
        method='lora',
        backend='trl'
    )

    config.set_lora_params(r=32, alpha=64)
    config.set_training_params(epochs=5, learning_rate=1e-4)

    full_config = config.get_full_config()
    report("Config has model section", 'model' in full_config)
    report("Config has lora section", 'lora' in full_config)
    report("LoRA r updated", full_config['lora']['r'] == 32)
    report("Epochs updated", full_config['training']['num_train_epochs'] == 5)

    # Export
    export_dir = os.path.join(TEST_DIR, 'training_output')
    files = config.export(export_dir)

    report("Config JSON exported", os.path.exists(files['config']))
    report("Train script exported", os.path.exists(files['script']))
    report("Requirements exported", os.path.exists(files['requirements']))

    # Validate generated script
    with open(files['script']) as f:
        script = f.read()
    report("Script has model name", 'Meta-Llama-3-8B' in script)
    report("Script has LoRA config", 'LoraConfig' in script)
    report("Script has SFTTrainer", 'SFTTrainer' in script)


def test_layer6_monitoring():
    """Test Layer 6b: LLM Monitoring."""
    print("\n── Layer 6b: LLM Monitoring ──")
    from data_pipeline.llm_monitor import LLMMonitor

    log_dir = os.path.join(TEST_DIR, 'logs')
    monitor = LLMMonitor(log_dir)

    # Log runs
    monitor.log_run("v1.0", {
        'loss': 0.8, 'perplexity': 15.0,
        'bleu': 0.25, 'rouge_1': 0.40
    }, model_name="llama-3-8b")

    monitor.log_run("v1.1", {
        'loss': 0.5, 'perplexity': 10.0,
        'bleu': 0.35, 'rouge_1': 0.52
    }, model_name="llama-3-8b")

    # List
    runs = monitor.list_runs()
    report("2 runs logged", len(runs) == 2)

    # Compare
    comparison = monitor.compare_runs("v1.0", "v1.1")
    report("Comparison has improvements", len(comparison['improvements']) > 0)
    report("Overall improved", comparison['overall_improved'])

    # Retraining check
    trigger = monitor.check_retraining_triggers("v1.1")
    report("Trigger check works", 'should_retrain' in trigger)

    # Metric history
    history = monitor.get_metric_history('loss')
    report("History has 2 entries", len(history) == 2)

    # Report
    report_md = monitor.generate_report(
        os.path.join(TEST_DIR, 'eval_report.md')
    )
    report("Report generated", len(report_md) > 100)
    report("Report file saved", os.path.exists(os.path.join(TEST_DIR, 'eval_report.md')))


def test_full_pipeline(sources):
    """Test the full LLMPipeline orchestrator."""
    print("\n── Full Pipeline Orchestrator ──")
    from data_pipeline.llm_pipeline import LLMPipeline

    llm = LLMPipeline(
        registry_dir=os.path.join(TEST_DIR, 'pipeline_registry'),
        log_dir=os.path.join(TEST_DIR, 'pipeline_logs')
    )

    output_dir = os.path.join(TEST_DIR, 'pipeline_output')
    result = llm.run_full_pipeline(
        sources=sources,
        version='v1.0.0',
        output_dir=output_dir,
        model='meta-llama/Meta-Llama-3-8B',
        method='lora',
        domain='machine learning',
        template='alpaca',
        chunk_method='sliding_window',
        chunk_size=400,
        min_quality_score=0.3,
        description='Test pipeline run'
    )

    report("Pipeline produced data file", 'data' in result)
    report("Pipeline produced config", 'config' in result)
    report("Pipeline produced script", 'script' in result)
    report("Pipeline produced report", 'report' in result)

    # Verify data file
    if 'data' in result:
        with open(result['data']) as f:
            lines = [l for l in f if l.strip()]
        report("Training data has samples", len(lines) > 0, f"{len(lines)} samples")

    # Verify report
    pipeline_report = llm.get_report()
    report("Report has all stages",
           len(pipeline_report['stages_completed']) >= 5,
           f"stages: {pipeline_report['stages_completed']}")


def main():
    print("=" * 60)
    print("  LLM PIPELINE INTEGRATION TEST")
    print("=" * 60)
    print(f"  Test directory: {TEST_DIR}\n")

    try:
        sources = setup_test_files()
        print(f"  Created {len(sources)} test files\n")

        docs = test_layer1_ingestion(sources)
        chunks = test_layer2_chunking(docs)
        pairs = test_layer3_formatting(chunks)
        filtered = test_layer4_quality(pairs)
        test_layer5_versioning(filtered)
        test_layer6_finetune_config()
        test_layer6_monitoring()
        test_full_pipeline(sources)

    except Exception as e:
        print(f"\n{FAIL} FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Fatal Error", False))

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"  RESULTS: {passed}/{total} passed")

    if passed == total:
        print(f"  {PASS} ALL TESTS PASSED")
    else:
        failed = [(n, p) for n, p in results if not p]
        print(f"  {FAIL} {len(failed)} FAILED:")
        for name, _ in failed:
            print(f"     • {name}")

    print("=" * 60)

    # Cleanup
    try:
        shutil.rmtree(TEST_DIR)
        print(f"  🧹 Cleaned up: {TEST_DIR}")
    except Exception:
        print(f"  ⚠️  Manual cleanup needed: {TEST_DIR}")

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
