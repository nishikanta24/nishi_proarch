"""
API Dependencies Module
Handles singleton initialization of core AI and data components
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine

from ..ai.retriever import initialize_retriever, Retriever
from ..ai.qa_chain import QACopilot
from ..recommendations.recommendations import RecommendationEngine

logger = logging.getLogger(__name__)
load_dotenv()

# Global singletons (initialized once at startup)
_retriever: Optional[Retriever] = None
_qa_copilot: Optional[QACopilot] = None
_recommendation_engine: Optional[RecommendationEngine] = None
_db_engine = None

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/cost_analytics.db")


def get_db_engine():
    """
    Returns a SQLAlchemy engine for the database.
    Creates it once and reuses the same instance.
    """
    global _db_engine
    if _db_engine is None:
        _db_engine = create_engine(DATABASE_URL)
    return _db_engine


def initialize_app_dependencies():
    """
    Initialize all expensive resources once at application startup.
    This includes:
    - Retriever (with EmbeddingsManager, KPICalculator, AnomalyDetector)
    - QACopilot (LLM chain)
    - RecommendationEngine
    """
    global _retriever, _qa_copilot, _recommendation_engine
    
    logger.info("=" * 80)
    logger.info("INITIALIZING API DEPENDENCIES")
    logger.info("=" * 80)
    
    try:
        # Load configuration from environment
        database_url = os.getenv("DATABASE_URL", "sqlite:///./data/cost_analytics.db")
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
        reference_docs_path = os.getenv("REFERENCE_DOCS_PATH", "data/reference")
        
        logger.info(f"Database URL: {database_url}")
        logger.info(f"Vector Store Path: {vector_store_path}")
        logger.info(f"Reference Docs Path: {reference_docs_path}")
        
        # 1. Initialize Retriever (handles vector store + database queries)
        logger.info("Initializing Retriever...")
        _retriever = initialize_retriever(
            reference_docs_path=reference_docs_path,
            vector_store_path=vector_store_path,
            database_url=database_url,
            request_id="startup"
        )
        logger.info("✓ Retriever initialized")
        
        # 2. Initialize QACopilot (LLM chain)
        logger.info("Initializing QACopilot...")
        _qa_copilot = QACopilot(request_id="startup")
        logger.info("✓ QACopilot initialized")
        
        # 3. Initialize RecommendationEngine
        logger.info("Initializing RecommendationEngine...")
        _recommendation_engine = RecommendationEngine(
            engine=get_db_engine(),
            request_id="startup"
        )
        logger.info("✓ RecommendationEngine initialized")
        
        logger.info("=" * 80)
        logger.info("✓ ALL DEPENDENCIES INITIALIZED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Failed to initialize dependencies: {str(e)}", exc_info=True)
        raise


def get_retriever() -> Retriever:
    """Dependency injection for Retriever"""
    if _retriever is None:
        raise RuntimeError("Retriever not initialized. Call initialize_app_dependencies() first.")
    return _retriever


def get_qa_copilot() -> QACopilot:
    """Dependency injection for QACopilot"""
    if _qa_copilot is None:
        raise RuntimeError("QACopilot not initialized. Call initialize_app_dependencies() first.")
    return _qa_copilot


def get_recommendation_engine() -> RecommendationEngine:
    """Dependency injection for RecommendationEngine"""
    if _recommendation_engine is None:
        raise RuntimeError("RecommendationEngine not initialized. Call initialize_app_dependencies() first.")
    return _recommendation_engine