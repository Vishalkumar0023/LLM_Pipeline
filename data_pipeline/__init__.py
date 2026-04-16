"""
Data Pipeline Tool
==================
A comprehensive, modular Python tool for automated data cleaning,
exploratory data analysis (EDA), feature engineering, and
enterprise LLM fine-tuning data preparation.

Modules:
- data_loader: Load and validate datasets
- data_cleaner: Clean and preprocess data
- eda: Exploratory data analysis and visualizations
- feature_engineer: Feature engineering and transformation
- pipeline: Main orchestrator combining all modules
- document_ingestor: Multi-source document ingestion (PDF, URL, XML, text)
- text_chunker: Text chunking strategies for LLM training
- instruct_formatter: Instruction/response pair formatting (Alpaca, ChatML, ShareGPT)
- quality_scorer: Training data quality scoring and filtering
- dataset_registry: Dataset versioning and registry
- finetune_config: LoRA/SFT fine-tuning config generation
- llm_monitor: Model evaluation monitoring and retraining triggers
- llm_pipeline: End-to-end LLM data pipeline orchestrator
- evidence_builder: Layer A deterministic evidence parser
- dataset_generator: Layer B instruction dataset synthesis
- verification_agent: Dataset verifier/corrector
- pipeline_runner: Two-layer end-to-end dataset pipeline
"""

from .data_loader import DataLoader
from .data_cleaner import DataCleaner
from .model_trainer import ModelTrainer

# Lazy imports to avoid loading heavy dependencies at import time
# Use: from data_pipeline.eda import EDAAnalyzer
# Use: from data_pipeline.feature_engineer import FeatureEngineer
# Use: from data_pipeline.pipeline import DataPipeline
# Use: from data_pipeline.llm_pipeline import LLMPipeline


def _get_pipeline():
    from .pipeline import DataPipeline as _DP

    return _DP


# Make classes available but lazy
import importlib


def __getattr__(name):
    if name == "DataPipeline":
        return _get_pipeline()
    if name == "EDAAnalyzer":
        from .eda import EDAAnalyzer

        return EDAAnalyzer
    if name == "FeatureEngineer":
        from .feature_engineer import FeatureEngineer

        return FeatureEngineer
    # LLM Pipeline modules (lazy)
    if name == "LLMPipeline":
        from .llm_pipeline import LLMPipeline

        return LLMPipeline
    if name == "DocumentIngestor":
        from .document_ingestor import DocumentIngestor

        return DocumentIngestor
    if name == "TextChunker":
        from .text_chunker import TextChunker

        return TextChunker
    if name == "InstructFormatter":
        from .instruct_formatter import InstructFormatter

        return InstructFormatter
    if name == "QualityScorer":
        from .quality_scorer import QualityScorer

        return QualityScorer
    if name == "DatasetRegistry":
        from .dataset_registry import DatasetRegistry

        return DatasetRegistry
    if name == "FineTuneConfig":
        from .finetune_config import FineTuneConfig

        return FineTuneConfig
    if name == "LLMMonitor":
        from .llm_monitor import LLMMonitor

        return LLMMonitor
    if name == "EcommerceScraper":
        from .ecommerce_scraper import EcommerceScraper

        return EcommerceScraper
    if name == "EvidenceBuilder":
        from .evidence_builder import EvidenceBuilder

        return EvidenceBuilder
    if name == "DatasetGenerator":
        from .dataset_generator import DatasetGenerator

        return DatasetGenerator
    if name == "DatasetVerificationAgent":
        from .verification_agent import DatasetVerificationAgent

        return DatasetVerificationAgent
    if name == "TwoLayerPipelineRunner":
        from .pipeline_runner import TwoLayerPipelineRunner

        return TwoLayerPipelineRunner
    if name == "TwoLayerQualityScorer":
        from .quality_scorer import TwoLayerQualityScorer

        return TwoLayerQualityScorer
    if name == "EmbeddingEngine":
        from .embedding_engine import EmbeddingEngine

        return EmbeddingEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "2.1.0"
__all__ = [
    # Original modules
    "DataPipeline",
    "DataLoader",
    "DataCleaner",
    "EDAAnalyzer",
    "FeatureEngineer",
    "ModelTrainer",
    # LLM Pipeline modules
    "LLMPipeline",
    "DocumentIngestor",
    "TextChunker",
    "InstructFormatter",
    "QualityScorer",
    "DatasetRegistry",
    "FineTuneConfig",
    "LLMMonitor",
    "EcommerceScraper",
    # Two-layer architecture modules
    "EvidenceBuilder",
    "DatasetGenerator",
    "DatasetVerificationAgent",
    "TwoLayerQualityScorer",
    "TwoLayerPipelineRunner",
]
