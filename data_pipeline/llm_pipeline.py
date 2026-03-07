"""
LLM Pipeline Module
===================
Top-level orchestrator that chains all 6 enterprise layers
for domain-specific LLM fine-tuning data preparation.
"""

import os
from typing import List, Dict, Any, Optional, Union

from .document_ingestor import DocumentIngestor
from .text_chunker import TextChunker
from .instruct_formatter import InstructFormatter
from .quality_scorer import QualityScorer
from .dataset_registry import DatasetRegistry
from .finetune_config import FineTuneConfig
from .llm_monitor import LLMMonitor


class LLMPipeline:
    """
    End-to-end LLM fine-tuning data pipeline.

    Chains: Ingestion → Chunking → Formatting → Quality Scoring
            → Versioning → Training Config → Monitoring

    Example:
    --------
    >>> from data_pipeline import LLMPipeline
    >>>
    >>> llm = LLMPipeline(registry_dir="./llm_datasets")
    >>> llm.ingest(["paper.pdf", "https://docs.example.com", "notes.txt"])
    >>> llm.chunk(method="sliding_window", chunk_size=512, overlap=64)
    >>> llm.format_instructions(template="alpaca", domain="medical")
    >>> llm.score_quality(min_score=0.5)
    >>> llm.version_dataset("v1.0.0", description="Initial medical dataset")
    >>> llm.generate_training_config(model="meta-llama/Llama-3-8B", method="lora")
    >>> llm.export("./output")
    """

    def __init__(
        self, registry_dir: str = "./dataset_registry", log_dir: str = "./llm_logs"
    ):
        """
        Initialize the LLM Pipeline.

        Parameters
        ----------
        registry_dir : str
            Directory for dataset versioning.
        log_dir : str
            Directory for monitoring logs.
        """
        self.registry_dir = registry_dir
        self.log_dir = log_dir

        # Pipeline state
        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []
        self.pairs: List[Dict[str, Any]] = []
        self.scored_pairs: List[Dict[str, Any]] = []
        self.filtered_pairs: List[Dict[str, Any]] = []

        # Sub-modules (initialized lazily)
        self._ingestor: Optional[DocumentIngestor] = None
        self._chunker: Optional[TextChunker] = None
        self._formatter: Optional[InstructFormatter] = None
        self._scorer: Optional[QualityScorer] = None
        self._registry: Optional[DatasetRegistry] = None
        self._config: Optional[FineTuneConfig] = None
        self._monitor: Optional[LLMMonitor] = None

        # Pipeline report
        self._report: Dict[str, Any] = {"stages_completed": [], "errors": []}

    # ─── Stage 1: Ingestion ──────────────────────────────────────────────

    def ingest(
        self,
        sources: Union[str, List[str]],
        recursive: bool = False,
        encoding: str = "utf-8",
        timeout: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Ingest documents from multiple sources.

        Parameters
        ----------
        sources : str or list of str
            File paths, URLs, or directories.
        recursive : bool
            Recursively search directories.
        encoding : str
            Text file encoding.
        timeout : int
            HTTP timeout for URL scraping.

        Returns
        -------
        list of dict
            Ingested documents.
        """
        print("\n" + "=" * 60)
        print("📥 STAGE 1: DOCUMENT INGESTION")
        print("=" * 60)

        self._ingestor = DocumentIngestor(encoding=encoding, timeout=timeout)
        self.documents = self._ingestor.ingest(sources, recursive=recursive)
        self._ingestor.print_summary()

        self._report["stages_completed"].append("ingestion")
        self._report["ingestion"] = self._ingestor.get_stats()

        return self.documents

    def ingest_ecommerce(
        self,
        urls: Union[str, List[str]],
        use_playwright: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Ingest product data from e-commerce URLs.

        Parameters
        ----------
        urls : str or list of str
            Product URLs.
        use_playwright : bool
            Use Playwright for JS rendering.

        Returns
        -------
        list of dict
            Ingested product documents.
        """
        print("\n" + "=" * 60)
        print("📥 STAGE 1: E-COMMERCE INGESTION")
        print("=" * 60)

        from .ecommerce_scraper import EcommerceScraper
        scraper = EcommerceScraper(use_playwright=use_playwright, headless=True)
        self.documents = scraper.scrape_to_documents(urls)
        scraper.print_summary()

        self._report["stages_completed"].append("ecommerce_ingestion")
        self._report["ecommerce_ingestion"] = scraper.get_stats()

        return self.documents

    def ingest_ecommerce_listings(
        self,
        urls: Union[str, List[str]],
        max_pages: int = 10,
        use_playwright: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Ingest product data from e-commerce search/listing pages
        with pagination support (up to max_pages pages per URL).

        Parameters
        ----------
        urls : str or list of str
            Search or category listing URLs.
        max_pages : int
            Maximum pages to scrape per URL (default 10).
        use_playwright : bool
            Use Playwright for JS rendering.

        Returns
        -------
        list of dict
            Ingested product documents from all pages.
        """
        print("\n" + "=" * 60)
        print("📥 STAGE 1: E-COMMERCE LISTING INGESTION (PAGINATED)")
        print("=" * 60)

        from .ecommerce_scraper import EcommerceScraper
        scraper = EcommerceScraper(
            use_playwright=use_playwright, headless=True, max_pages=max_pages
        )
        self.documents = scraper.scrape_listings_to_documents(urls, max_pages=max_pages)
        scraper.print_summary()

        self._report["stages_completed"].append("ecommerce_listing_ingestion")
        self._report["ecommerce_listing_ingestion"] = scraper.get_stats()

        return self.documents

    # ─── Stage 2: Chunking ───────────────────────────────────────────────

    def chunk(
        self,
        method: str = "paragraph",
        chunk_size: int = 512,
        overlap: int = 64,
        min_chunk_size: int = 50,
        max_chunk_size: int = 2048,
    ) -> List[Dict[str, Any]]:
        """
        Chunk ingested documents.

        Parameters
        ----------
        method : str
            Chunking strategy: 'sentence', 'paragraph', 'sliding_window', 'token_aware'.
        chunk_size : int
            Target chunk size.
        overlap : int
            Overlap between consecutive chunks.
        min_chunk_size : int
            Minimum chunk size.
        max_chunk_size : int
            Maximum chunk size.

        Returns
        -------
        list of dict
            Text chunks.
        """
        if not self.documents:
            raise ValueError("No documents to chunk. Run ingest() first.")

        print("\n" + "=" * 60)
        print("🧩 STAGE 2: TEXT CHUNKING")
        print("=" * 60)

        self._chunker = TextChunker(
            method=method,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )
        self.chunks = self._chunker.chunk_documents(self.documents)
        self._chunker.print_summary()

        self._report["stages_completed"].append("chunking")
        self._report["chunking"] = self._chunker.get_stats()

        return self.chunks

    # ─── Stage 3: Instruction Formatting ─────────────────────────────────

    def format_instructions(
        self,
        template: str = "alpaca",
        domain: str = "general",
        generate_qa: bool = True,
        pairs_per_chunk: int = 2,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Format chunks into instruction-response pairs.

        Parameters
        ----------
        template : str
            Output format: 'alpaca', 'chatml', 'sharegpt'.
        domain : str
            Domain label for instruction generation.
        generate_qa : bool
            Generate Q&A pairs from content.
        pairs_per_chunk : int
            Number of pairs per chunk.
        system_prompt : str, optional
            System prompt for ChatML/ShareGPT.

        Returns
        -------
        list of dict
            Formatted training pairs.
        """
        if not self.chunks:
            raise ValueError("No chunks available. Run chunk() first.")

        print("\n" + "=" * 60)
        print("📝 STAGE 3: INSTRUCTION FORMATTING")
        print("=" * 60)

        self._formatter = InstructFormatter(
            template=template, system_prompt=system_prompt
        )
        self.pairs = self._formatter.format_chunks(
            self.chunks,
            domain=domain,
            generate_qa=generate_qa,
            pairs_per_chunk=pairs_per_chunk,
        )
        self._formatter.print_summary()

        self._report["stages_completed"].append("formatting")
        self._report["formatting"] = self._formatter.get_stats()

        return self.pairs

    # ─── Stage 3.5: LLM-Based Generation (Alternative to 2 & 3) ─────────

    def generate_with_llm(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ) -> List[Dict[str, Any]]:
        """
        Alternative to chunk() and format_instructions().
        Uses an LLM (OpenAI, Ollama, etc) to analyze full documents, clean the text,
        extract JSON properties, and generate highly-accurate, validated QA pairs.
        
        This reads directly from `self.documents` and sets `self.pairs`.
        """
        if not self.documents:
            raise ValueError("No documents available. Run ingest() first.")

        print("\n" + "=" * 60)
        print("🤖 STAGE 3.5: LLM-BASED INSTRUCTION GENERATION")
        print("=" * 60)
        
        try:
            from .llm_client import LLMClient
            from .llm_data_processor import LLMDataProcessor
        except ImportError:
            raise ImportError("LLM Data Processor modules not found.")

        client = LLMClient(provider=provider, api_key=api_key, base_url=base_url)
        processor = LLMDataProcessor(client=client, model=model)
        
        all_pairs = []
        for i, doc in enumerate(self.documents):
            print(f"  Processing document {i+1}/{len(self.documents)} via LLM...")
            
            # Combine the raw text from chunks if they exist, or use raw text
            text = doc.get("text", "")
            if not text:
                continue
                
            # Run the 5-stage cleaning pipeline
            valid_pairs = processor.process_raw_text(text)
            
            # Format to target structure with metadata
            for pair in valid_pairs:
                pair["metadata"] = {
                    "source": doc.get("metadata", {}).get("source", "unknown"),
                    "doc_id": doc.get("doc_id", "unknown"),
                    "generated_by": f"{provider}/{model}",
                    "is_ecommerce": doc.get("source_type") == "ecommerce"
                }
                all_pairs.append(pair)
                
        self.pairs = all_pairs
        
        # Merge processor stats into pipeline report
        stats = processor.get_stats()
        print("\nLLM Generation Summary:")
        print(f"  Docs Processed: {stats['processed']}")
        print(f"  Pairs Generated: {stats['qa_generated']}")
        print(f"  Passed Validation: {stats['validated_valid']}")
        print(f"  Failed Validation: {stats['validated_invalid']}")
        
        self._report["stages_completed"].append("llm_generation")
        self._report["llm_generation"] = stats
        
        return self.pairs

    # ─── Stage 4: Quality Scoring ────────────────────────────────────────

    def score_quality(
        self,
        min_score: float = 0.4,
        similarity_threshold: float = 0.85,
        toxic_keywords: Optional[set] = None,
        min_length: int = 50,
        max_length: int = 10000,
    ) -> List[Dict[str, Any]]:
        """
        Score and filter training pairs for quality.

        Parameters
        ----------
        min_score : float
            Minimum quality score to keep.
        similarity_threshold : float
            Cosine similarity threshold for dedup.
        toxic_keywords : set, optional
            Custom toxicity keywords.
        min_length : int
            Min text length in chars.
        max_length : int
            Max text length in chars.

        Returns
        -------
        list of dict
            Filtered, quality-scored pairs.
        """
        if not self.pairs:
            raise ValueError("No pairs to score. Run format_instructions() first.")

        print("\n" + "=" * 60)
        print("📊 STAGE 4: QUALITY SCORING")
        print("=" * 60)

        self._scorer = QualityScorer(
            min_quality_score=min_score,
            similarity_threshold=similarity_threshold,
            toxic_keywords=toxic_keywords,
            min_length=min_length,
            max_length=max_length,
        )
        self.scored_pairs = self._scorer.score(self.pairs)
        self.filtered_pairs = self._scorer.filter(self.scored_pairs, min_score)
        self._scorer.print_summary()

        self._report["stages_completed"].append("quality_scoring")
        self._report["quality_scoring"] = self._scorer.get_stats()

        return self.filtered_pairs

    # ─── Stage 5: Dataset Versioning ─────────────────────────────────────

    def version_dataset(
        self,
        version: str,
        description: str = "",
        source_files: Optional[List[str]] = None,
        parent_version: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Register the current dataset as a versioned snapshot.

        Parameters
        ----------
        version : str
            Semantic version (e.g., "v1.0.0").
        description : str
            Version description.
        source_files : list, optional
            Source files used.
        parent_version : str, optional
            Parent version.
        tags : list, optional
            Version tags.

        Returns
        -------
        dict
            Version metadata.
        """
        data = self.filtered_pairs if self.filtered_pairs else self.pairs
        if not data:
            raise ValueError("No data to version. Run previous stages first.")

        print("\n" + "=" * 60)
        print("📦 STAGE 5: DATASET VERSIONING")
        print("=" * 60)

        self._registry = DatasetRegistry(self.registry_dir)
        metadata = self._registry.register(
            data=data,
            version=version,
            description=description,
            source_files=source_files,
            parent_version=parent_version,
            tags=tags,
        )

        self._report["stages_completed"].append("versioning")
        self._report["versioning"] = metadata

        return metadata

    # ─── Stage 6: Training Config Generation ─────────────────────────────

    def generate_training_config(
        self,
        model: str = "meta-llama/Meta-Llama-3-8B",
        method: str = "lora",
        backend: str = "trl",
        lora_params: Optional[Dict] = None,
        training_params: Optional[Dict] = None,
    ) -> "FineTuneConfig":
        """
        Generate training configuration and scripts.

        Parameters
        ----------
        model : str
            HuggingFace model name.
        method : str
            Fine-tuning method: 'lora', 'qlora', 'full'.
        backend : str
            Backend: 'trl' or 'unsloth'.
        lora_params : dict, optional
            LoRA parameter overrides.
        training_params : dict, optional
            Training parameter overrides.

        Returns
        -------
        FineTuneConfig
            The config object (call .export() to save).
        """
        print("\n" + "=" * 60)
        print("⚙️  STAGE 6: TRAINING CONFIGURATION")
        print("=" * 60)

        self._config = FineTuneConfig(model_name=model, method=method, backend=backend)

        if lora_params:
            self._config.set_lora_params(**lora_params)
        if training_params:
            self._config.set_training_params(**training_params)

        self._config.print_summary()

        self._report["stages_completed"].append("training_config")
        self._report["training_config"] = self._config.get_full_config()

        return self._config

    # ─── Export ───────────────────────────────────────────────────────────

    def export(
        self,
        output_dir: str,
        include_config: bool = True,
        include_data: bool = True,
        include_metadata: bool = False,
    ) -> Dict[str, str]:
        """
        Export all pipeline outputs to a directory.

        Parameters
        ----------
        output_dir : str
            Output directory path.
        include_config : bool
            Export training config and script.
        include_data : bool
            Export training data JSONL.
        include_metadata : bool
            Include metadata in data export.

        Returns
        -------
        dict
            Paths to generated files.
        """
        os.makedirs(output_dir, exist_ok=True)
        result = {}

        print("\n" + "=" * 60)
        print("💾 EXPORTING PIPELINE OUTPUTS")
        print("=" * 60)

        # Export training data
        if include_data and self._formatter:
            data = self.filtered_pairs if self.filtered_pairs else self.pairs
            if data:
                data_path = os.path.join(output_dir, "training_data.jsonl")
                self._formatter.export_jsonl(
                    data, data_path, include_metadata=include_metadata
                )
                result["data"] = data_path
                print(f"📄 Training data: {data_path} ({len(data)} samples)")

        # Export training config and script
        if include_config and self._config:
            config_files = self._config.export(
                output_dir, dataset_path=result.get("data", "./training_data.jsonl")
            )
            result.update(config_files)

        # Export FAISS embeddings if available
        if self._scorer and getattr(self._scorer, "_embedding_engine", None):
            try:
                embed_dir = os.path.join(output_dir, "embeddings")
                embed_files = self._scorer._embedding_engine.save_index(embed_dir)
                result["embeddings_index"] = embed_files["index"]
                result["embeddings_data"] = embed_files["data"]
                print(f"🧠 Vector embeddings saved: {embed_files['index']}")
            except Exception as e:
                print(f"⚠️ Could not save vector embeddings: {e}")

        # Export pipeline report
        report_path = os.path.join(output_dir, "pipeline_report.json")
        import json

        with open(report_path, "w") as f:
            # Filter out non-serializable objects
            clean_report = {}
            for k, v in self._report.items():
                try:
                    json.dumps(v)
                    clean_report[k] = v
                except (TypeError, ValueError):
                    clean_report[k] = str(v)
            json.dump(clean_report, f, indent=2, default=str)
        result["report"] = report_path

        print(f"\n✅ All outputs saved to: {output_dir}")
        return result

    # ─── Monitoring ──────────────────────────────────────────────────────

    def log_evaluation(
        self, run_id: str, metrics: Dict[str, float], **kwargs
    ) -> Dict[str, Any]:
        """Log an evaluation run to the monitor."""
        self._monitor = LLMMonitor(self.log_dir)
        return self._monitor.log_run(run_id, metrics, **kwargs)

    def check_retraining(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Check if retraining is recommended."""
        if not self._monitor:
            self._monitor = LLMMonitor(self.log_dir)
        return self._monitor.check_retraining_triggers(run_id)

    def get_monitor(self) -> LLMMonitor:
        """Get the monitor instance for advanced operations."""
        if not self._monitor:
            self._monitor = LLMMonitor(self.log_dir)
        return self._monitor

    def get_registry(self) -> DatasetRegistry:
        """Get the registry instance for advanced operations."""
        if not self._registry:
            self._registry = DatasetRegistry(self.registry_dir)
        return self._registry

    # ─── Convenience ─────────────────────────────────────────────────────

    def run_ecommerce_pipeline(
        self,
        urls: Union[str, List[str]],
        version: str = "v1.0.0",
        output_dir: str = "./llm_output",
        model: str = "meta-llama/Meta-Llama-3-8B",
        method: str = "lora",
        template: str = "alpaca",
        chunk_method: str = "paragraph",
        chunk_size: int = 1024,
        min_quality_score: float = 0.6,
        use_playwright: bool = False,
        description: str = "",
    ) -> Dict[str, str]:
        """
        Run the complete pipeline optimized for e-commerce product data.

        Uses higher quality thresholds and e-commerce-aware formatting
        to produce clean, reliable instruction-output pairs from
        scraped product pages.

        Parameters
        ----------
        urls : str or list of str
            E-commerce product URLs.
        version : str
            Dataset version.
        output_dir : str
            Output directory.
        model : str
            Target LLM model name.
        method : str
            Fine-tuning method: 'lora', 'qlora', 'full'.
        template : str
            Output format: 'alpaca', 'chatml', 'sharegpt'.
        chunk_method : str
            Chunking strategy (default 'paragraph' for product text).
        chunk_size : int
            Target chunk size (default 1024 for richer context).
        min_quality_score : float
            Minimum quality score (default 0.6, higher than generic).
        use_playwright : bool
            Use Playwright for JS rendering.
        description : str
            Dataset version description.

        Returns
        -------
        dict
            Paths to all generated files.
        """
        print("\n" + "🛒" * 20)
        print("  E-COMMERCE LLM PIPELINE — FULL RUN")
        print("🛒" * 20)

        # Stage 1: E-commerce Ingest
        self.ingest_ecommerce(urls, use_playwright=use_playwright)

        # Stage 2: Chunk
        self.chunk(method=chunk_method, chunk_size=chunk_size)

        # Stage 3: Format with e-commerce domain
        self.format_instructions(
            template=template, domain="ecommerce", pairs_per_chunk=3
        )

        # Stage 4: Quality scoring with higher threshold
        self.score_quality(min_score=min_quality_score)

        # Stage 5: Version
        self.version_dataset(version=version, description=description)

        # Stage 6: Config
        self.generate_training_config(model=model, method=method)

        # Export
        result = self.export(output_dir)

        self._print_final_summary()

        return result

    def run_full_pipeline(
        self,
        sources: Union[str, List[str]],
        version: str = "v1.0.0",
        output_dir: str = "./llm_output",
        model: str = "meta-llama/Meta-Llama-3-8B",
        method: str = "lora",
        domain: str = "general",
        template: str = "alpaca",
        chunk_method: str = "sliding_window",
        chunk_size: int = 512,
        min_quality_score: float = 0.4,
        description: str = "",
    ) -> Dict[str, str]:
        """
        Run the complete pipeline end-to-end.

        Parameters
        ----------
        sources : str or list of str
            Input sources (files, URLs, directories).
        version : str
            Dataset version to register.
        output_dir : str
            Output directory for all artifacts.
        model : str
            Target LLM model name.
        method : str
            Fine-tuning method: 'lora', 'qlora', 'full'.
        domain : str
            Domain label.
        template : str
            Output format: 'alpaca', 'chatml', 'sharegpt'.
        chunk_method : str
            Chunking strategy.
        chunk_size : int
            Target chunk size.
        min_quality_score : float
            Minimum quality score.
        description : str
            Dataset version description.

        Returns
        -------
        dict
            Paths to all generated files.
        """
        print("\n" + "🚀" * 20)
        print("  LLM DATA PIPELINE — FULL RUN")
        print("🚀" * 20)

        # Stage 1: Ingest
        self.ingest(sources)

        # Stage 2: Chunk
        self.chunk(method=chunk_method, chunk_size=chunk_size)

        # Stage 3: Format
        self.format_instructions(template=template, domain=domain)

        # Stage 4: Quality
        self.score_quality(min_score=min_quality_score)

        # Stage 5: Version
        self.version_dataset(version=version, description=description)

        # Stage 6: Config
        self.generate_training_config(model=model, method=method)

        # Export
        result = self.export(output_dir)

        self._print_final_summary()

        return result

    def _print_final_summary(self):
        """Print final pipeline summary."""
        print("\n" + "=" * 60)
        print("🏁 PIPELINE COMPLETE")
        print("=" * 60)
        print(f"\n📋 Stages completed: {len(self._report['stages_completed'])}")
        for stage in self._report["stages_completed"]:
            print(f"   ✅ {stage}")

        if self.documents:
            print(f"\n📥 Documents: {len(self.documents)}")
        if self.chunks:
            print(f"🧩 Chunks: {len(self.chunks)}")
        if self.pairs:
            print(f"📝 Pairs generated: {len(self.pairs)}")
        if self.filtered_pairs:
            print(f"✅ Pairs after filtering: {len(self.filtered_pairs)}")

        print("=" * 60)

    def get_report(self) -> Dict[str, Any]:
        """Return the full pipeline report."""
        return self._report
