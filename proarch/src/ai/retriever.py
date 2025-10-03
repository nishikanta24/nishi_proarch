"""
Retriever Module
Handles ALL retrieval operations for the RAG pipeline:
- Vector store semantic search (via EmbeddingsManager)
- Database queries for structured cost data
- Context assembly and formatting
- Recall@k calculation for RAG evaluation
"""

import os
import logging
import pandas as pd
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine

from .embeddings import EmbeddingsManager
from ..transformations.kpis import KPICalculator, get_available_months
from ..transformations.anomalies import AnomalyDetector
from ..transformations.aggregations import (
    get_monthly_costs,
    get_costs_by_service,
    get_costs_by_resource_group,
    get_top_n_resources,
    get_monthly_trend,
    get_cost_breakdown,
    get_unit_cost_changes,
    get_resources_without_tags
)

logger = logging.getLogger(__name__)


class Retriever:
    """
    Unified retrieval system for RAG pipeline.
    Combines semantic search (vector store) with structured data queries (database).
    """
    
    def __init__(
        self,
        embeddings_manager: EmbeddingsManager,
        database_url: str = "sqlite:///./data/cost_analytics.db",
        request_id: str = "default"
    ):
        """
        Initialize the Retriever
        
        Args:
            embeddings_manager: Initialized EmbeddingsManager instance
            database_url: Database connection string
            request_id: Request ID for logging
        """
        self.embeddings_manager = embeddings_manager
        self.database_url = database_url
        self.request_id = request_id
        
        # Initialize database engine
        self.engine = create_engine(database_url)
        
        # Initialize KPI calculator and anomaly detector
        self.kpi_calculator = KPICalculator(self.engine, request_id)
        self.anomaly_detector = AnomalyDetector(self.engine, request_id)
        
        logger.info(
            f"Retriever initialized with database: {database_url}",
            extra={'request_id': request_id}
        )

        # Initialize caching system
        self._cache = {}
        self._cache_ttl = timedelta(minutes=int(os.getenv("CACHE_TTL_MINUTES", "30")))  # 30 min default
        self._cache_enabled = os.getenv("ENABLE_CACHE", "true").lower() == "true"

        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0

        if self._cache_enabled:
            logger.info(
                f"Cache enabled with TTL: {self._cache_ttl}",
                extra={'request_id': request_id}
            )

    def _get_cache_key(self, question: str, k: int, include_structured_data: bool) -> str:
        """
        Generate a cache key for the given retrieval parameters.

        Args:
            question: The user's question
            k: Number of documents to retrieve
            include_structured_data: Whether structured data is included

        Returns:
            str: Cache key (SHA256 hash)
        """
        cache_data = {
            'question': question.lower().strip(),
            'k': k,
            'include_structured_data': include_structured_data
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """
        Retrieve a cached result if it exists and hasn't expired.

        Args:
            cache_key: The cache key to look up

        Returns:
            dict or None: Cached result or None if not found/expired
        """
        if not self._cache_enabled or cache_key not in self._cache:
            return None

        cached_item = self._cache[cache_key]
        if datetime.now() > cached_item['expires_at']:
            # Cache expired, remove it
            del self._cache[cache_key]
            return None

        # Cache hit
        self.cache_hits += 1
        logger.info(
            f"Cache hit for key: {cache_key[:8]}...",
            extra={'request_id': self.request_id}
        )
        return cached_item['result']

    def _set_cached_result(self, cache_key: str, result: Dict) -> None:
        """
        Store a result in the cache with TTL.

        Args:
            cache_key: The cache key
            result: The result to cache
        """
        if not self._cache_enabled:
            return

        self._cache[cache_key] = {
            'result': result,
            'cached_at': datetime.now(),
            'expires_at': datetime.now() + self._cache_ttl
        }

        # Clean up expired entries periodically (every 10 cache sets)
        if len(self._cache) % 10 == 0:
            self._cleanup_expired_cache()

    def _cleanup_expired_cache(self) -> None:
        """Remove expired cache entries."""
        current_time = datetime.now()
        expired_keys = [
            key for key, item in self._cache.items()
            if current_time > item['expires_at']
        ]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.info(
                f"Cleaned up {len(expired_keys)} expired cache entries",
                extra={'request_id': self.request_id}
            )

    def get_cache_metrics(self) -> Dict:
        """
        Get cache performance metrics.

        Returns:
            dict: Cache metrics including hits, misses, hit rate
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            'cache_enabled': self._cache_enabled,
            'cache_ttl_minutes': self._cache_ttl.total_seconds() / 60,
            'cache_size': len(self._cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate_percent': round(hit_rate, 2)
        }

    def retrieve(
        self,
        question: str,
        k: int = 5,
        include_structured_data: bool = True
    ) -> Dict:
        """
        Main retrieval method - combines semantic and structured retrieval

        Args:
            question: User's natural language question
            k: Number of documents to retrieve from vector store
            include_structured_data: Whether to query database for analytical data

        Returns:
            dict: Complete retrieval result with context, sources, and metadata
        """
        logger.info(
            f"Retrieving context for question: '{question[:100]}...' (k={k})",
            extra={'request_id': self.request_id}
        )

        retrieval_start = datetime.now()

        # Check cache first
        cache_key = self._get_cache_key(question, k, include_structured_data)
        cached_result = self._get_cached_result(cache_key)

        if cached_result:
            # Return cached result with updated metrics
            cached_result_copy = cached_result.copy()
            cached_result_copy['cache_metrics'] = self.get_cache_metrics()
            cached_result_copy['retrieval_time_seconds'] = round(
                (datetime.now() - retrieval_start).total_seconds(), 3
            )
            return cached_result_copy

        # Cache miss - proceed with normal retrieval
        self.cache_misses += 1
        
        try:
            # 1. Semantic Retrieval (Vector Store)
            semantic_results = self._semantic_retrieval(question, k)
            
            # 2. Intent Classification & Structured Retrieval
            structured_results = {}
            if include_structured_data:
                intent = self._classify_intent(question)
                structured_results = self._structured_retrieval(question, intent)
            
            # 3. Assemble Context
            context = self._assemble_context(semantic_results, structured_results)
            
            # 4. Calculate metrics
            retrieval_duration = (datetime.now() - retrieval_start).total_seconds()
            
            # 5. Build response
            response = {
                'status': 'success',
                'request_id': self.request_id,
                'question': question,
                'context': context,
                'retrieval_time_seconds': round(retrieval_duration, 3),
                'semantic_results': {
                    'documents_retrieved': len(semantic_results['documents']),
                    'sources': semantic_results['sources']
                },
                'structured_results': {
                    'queries_executed': structured_results.get('queries_executed', 0),
                    'data_points': structured_results.get('data_points', 0)
                },
                'cache_metrics': self.get_cache_metrics(),
                'metadata': {
                    'vector_store_k': k,
                    'include_structured_data': include_structured_data,
                    'intent': structured_results.get('intent', 'unknown')
                }
            }
            
            logger.info(
                f"✓ Retrieval complete in {retrieval_duration:.3f}s: "
                f"{len(semantic_results['documents'])} docs, "
                f"{structured_results.get('data_points', 0)} data points",
                extra={'request_id': self.request_id}
            )

            # Cache the result for future requests
            self._set_cached_result(cache_key, response)

            return response
            
        except Exception as e:
            logger.error(
                f"Error during retrieval: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def _semantic_retrieval(self, question: str, k: int) -> Dict:
        """
        Perform semantic search against vector store
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            dict: Retrieved documents and metadata
        """
        logger.info(
            f"Performing semantic retrieval (k={k})",
            extra={'request_id': self.request_id}
        )
        
        try:
            # Perform similarity search
            documents = self.embeddings_manager.similarity_search(
                query=question,
                k=k
            )
            
            # Extract sources
            sources = list(set([doc.metadata.get('source', 'unknown') for doc in documents]))
            
            # Format document text
            doc_texts = []
            for i, doc in enumerate(documents):
                source = doc.metadata.get('source', 'unknown')
                doc_texts.append(f"[Document {i+1} from {source}]:\n{doc.page_content}\n")
            
            return {
                'documents': documents,
                'sources': sources,
                'formatted_texts': doc_texts
            }
            
        except Exception as e:
            logger.error(
                f"Error in semantic retrieval: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            # Return empty results on failure
            return {
                'documents': [],
                'sources': [],
                'formatted_texts': []
            }
    
    def _classify_intent(self, question: str) -> str:
        """
        Simple keyword-based intent classification
        
        Args:
            question: User's question
            
        Returns:
            str: Intent category
        """
        question_lower = question.lower()
        
        # Cost analysis intents
        if any(word in question_lower for word in ['cost', 'spend', 'bill', 'expense', 'price']):
            if any(word in question_lower for word in ['trend', 'over time', 'history', 'growth']):
                return 'cost_trend'
            elif any(word in question_lower for word in ['breakdown', 'by service', 'by resource']):
                return 'cost_breakdown'
            elif any(word in question_lower for word in ['top', 'most expensive', 'highest']):
                return 'top_costs'
            else:
                return 'cost_query'
        
        # Anomaly detection intents
        if any(word in question_lower for word in ['anomaly', 'anomalies', 'spike', 'unusual', 'unexpected']):
            return 'anomaly_detection'
        
        # Tagging compliance intents
        if any(word in question_lower for word in ['tag', 'tagged', 'untagged', 'compliance']):
            return 'tagging_compliance'
        
        # Recommendation intents
        if any(word in question_lower for word in ['recommend', 'suggestion', 'optimize', 'save']):
            return 'recommendations'
        
        # Default to general query
        return 'general_query'
    
    def _structured_retrieval(self, question: str, intent: str) -> Dict:
        """
        Query database for structured cost data based on intent
        
        Args:
            question: User's question
            intent: Classified intent
            
        Returns:
            dict: Structured data results
        """
        logger.info(
            f"Performing structured retrieval (intent: {intent})",
            extra={'request_id': self.request_id}
        )
        
        try:
            # Extract month from question (default to latest available)
            month = self._extract_month_from_question(question)
            if not month:
                available_months = get_available_months(self.engine, self.request_id)
                month = available_months[0] if available_months else None
            
            if not month:
                logger.warning(
                    "No month data available for structured retrieval",
                    extra={'request_id': self.request_id}
                )
                return {
                    'intent': intent,
                    'queries_executed': 0,
                    'data_points': 0,
                    'data': {}
                }
            
            # Get previous month for comparisons
            month_dt = datetime.strptime(month, '%Y-%m')
            prev_month_dt = month_dt - relativedelta(months=1)
            prev_month = prev_month_dt.strftime('%Y-%m')
            
            results = {
                'intent': intent,
                'month': month,
                'queries_executed': 0,
                'data_points': 0,
                'data': {}
            }
            
            # Execute queries based on intent
            if intent == 'cost_query' or intent == 'top_costs':
                # Monthly costs
                monthly_costs = get_monthly_costs(self.engine, month, self.request_id)
                if not monthly_costs.empty:
                    results['data']['monthly_cost'] = monthly_costs.to_dict('records')
                    results['queries_executed'] += 1
                    results['data_points'] += len(monthly_costs)
                
                # Top resources
                top_resources = get_top_n_resources(self.engine, month, n=10, request_id=self.request_id)
                if not top_resources.empty:
                    results['data']['top_resources'] = top_resources.to_dict('records')
                    results['queries_executed'] += 1
                    results['data_points'] += len(top_resources)
            
            elif intent == 'cost_breakdown':
                # Service breakdown
                service_costs = get_costs_by_service(self.engine, month, self.request_id)
                if not service_costs.empty:
                    results['data']['service_costs'] = service_costs.head(10).to_dict('records')
                    results['queries_executed'] += 1
                    results['data_points'] += len(service_costs)
                
                # Resource group breakdown
                rg_costs = get_costs_by_resource_group(self.engine, month, self.request_id)
                if not rg_costs.empty:
                    results['data']['resource_group_costs'] = rg_costs.head(10).to_dict('records')
                    results['queries_executed'] += 1
                    results['data_points'] += len(rg_costs)
            
            elif intent == 'cost_trend':
                # Monthly trend
                trend_data = get_monthly_trend(self.engine, months=6, request_id=self.request_id)
                if not trend_data.empty:
                    results['data']['trend'] = trend_data.to_dict('records')
                    results['queries_executed'] += 1
                    results['data_points'] += len(trend_data)
            
            elif intent == 'anomaly_detection':
                # Detect anomalies
                anomalies = self.anomaly_detector.detect_all_anomalies(month, prev_month)
                results['data']['anomalies'] = anomalies
                results['queries_executed'] += 5  # Multiple anomaly detection queries
                results['data_points'] += anomalies.get('summary', {}).get('total_anomalies', 0)
            
            elif intent == 'tagging_compliance':
                # Tagging compliance
                compliance = self.kpi_calculator.get_tagging_compliance_kpi(month)
                results['data']['tagging_compliance'] = compliance
                results['queries_executed'] += 1
                results['data_points'] += compliance.get('summary', {}).get('untagged_resources', 0)
            
            else:
                # Default: Get comprehensive KPIs
                kpis = self.kpi_calculator.calculate_monthly_kpis(month)
                results['data']['kpis'] = kpis
                results['queries_executed'] += 1
                results['data_points'] += len(kpis.get('top_services', []))
            
            logger.info(
                f"Structured retrieval complete: {results['queries_executed']} queries, "
                f"{results['data_points']} data points",
                extra={'request_id': self.request_id}
            )
            
            return results
            
        except Exception as e:
            logger.error(
                f"Error in structured retrieval: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            return {
                'intent': intent,
                'queries_executed': 0,
                'data_points': 0,
                'data': {},
                'error': str(e)
            }
    
    def _extract_month_from_question(self, question: str) -> Optional[str]:
        """
        Extract month from question using keyword matching
        
        Args:
            question: User's question
            
        Returns:
            Optional[str]: Month in YYYY-MM format, or None
        """
        question_lower = question.lower()
        
        # Month name mapping
        month_names = {
            'january': '01', 'jan': '01',
            'february': '02', 'feb': '02',
            'march': '03', 'mar': '03',
            'april': '04', 'apr': '04',
            'may': '05',
            'june': '06', 'jun': '06',
            'july': '07', 'jul': '07',
            'august': '08', 'aug': '08',
            'september': '09', 'sep': '09',
            'october': '10', 'oct': '10',
            'november': '11', 'nov': '11',
            'december': '12', 'dec': '12'
        }
        
        # Try to find month name and year
        import re
        for month_name, month_num in month_names.items():
            if month_name in question_lower:
                # Look for year near month name
                year_match = re.search(r'(20\d{2})', question)
                if year_match:
                    year = year_match.group(1)
                    return f"{year}-{month_num}"
        
        # Try to find YYYY-MM format directly
        date_match = re.search(r'(20\d{2})[-/](0?[1-9]|1[0-2])', question)
        if date_match:
            year = date_match.group(1)
            month = date_match.group(2).zfill(2)
            return f"{year}-{month}"
        
        return None
    
    def _assemble_context(
        self,
        semantic_results: Dict,
        structured_results: Dict
    ) -> str:
        """
        Combine semantic and structured results into unified context string
        
        Args:
            semantic_results: Results from vector store
            structured_results: Results from database queries
            
        Returns:
            str: Formatted context for LLM
        """
        logger.info(
            "Assembling unified context from retrieval results",
            extra={'request_id': self.request_id}
        )
        
        context_parts = []
        
        # 1. Add semantic search results (reference documents)
        if semantic_results['formatted_texts']:
            context_parts.append("=== REFERENCE DOCUMENTS ===\n")
            context_parts.extend(semantic_results['formatted_texts'])
            context_parts.append("\n")
        
        # 2. Add structured data results
        if structured_results.get('data'):
            context_parts.append("=== BILLING DATA & ANALYTICS ===\n")
            
            data = structured_results['data']
            month = structured_results.get('month', 'N/A')
            
            # Format different data types
            if 'monthly_cost' in data:
                context_parts.append(f"\n[Monthly Cost Summary for {month}]:\n")
                context_parts.append(self._format_dataframe(pd.DataFrame(data['monthly_cost'])))
            
            if 'top_resources' in data:
                context_parts.append(f"\n[Top Cost Drivers for {month}]:\n")
                context_parts.append(self._format_dataframe(pd.DataFrame(data['top_resources'])))
            
            if 'service_costs' in data:
                context_parts.append(f"\n[Costs by Service for {month}]:\n")
                context_parts.append(self._format_dataframe(pd.DataFrame(data['service_costs'])))
            
            if 'resource_group_costs' in data:
                context_parts.append(f"\n[Costs by Resource Group for {month}]:\n")
                context_parts.append(self._format_dataframe(pd.DataFrame(data['resource_group_costs'])))
            
            if 'trend' in data:
                context_parts.append("\n[Monthly Cost Trend]:\n")
                context_parts.append(self._format_dataframe(pd.DataFrame(data['trend'])))
            
            if 'anomalies' in data:
                context_parts.append(f"\n[Detected Anomalies]:\n")
                context_parts.append(self._format_anomalies(data['anomalies']))
            
            if 'tagging_compliance' in data:
                context_parts.append(f"\n[Tagging Compliance for {month}]:\n")
                context_parts.append(self._format_tagging_compliance(data['tagging_compliance']))
            
            if 'kpis' in data:
                context_parts.append(f"\n[Key Performance Indicators]:\n")
                context_parts.append(self._format_kpis(data['kpis']))
            
            context_parts.append("\n")
        
        # Combine all parts
        context = "".join(context_parts)
        
        logger.info(
            f"Context assembled: {len(context)} characters",
            extra={'request_id': self.request_id}
        )
        
        return context
    
    def _format_dataframe(self, df: pd.DataFrame, max_rows: int = 10) -> str:
        """Format DataFrame as readable text"""
        if df.empty:
            return "(No data available)\n"
        
        # Limit rows
        df = df.head(max_rows)
        
        # Convert to markdown-style table
        return df.to_markdown(index=False) + "\n"
    
    def _format_anomalies(self, anomalies_data: Dict) -> str:
        """Format anomaly detection results"""
        lines = []
        
        summary = anomalies_data.get('summary', {})
        lines.append(f"Total anomalies detected: {summary.get('total_anomalies', 0)}\n")
        
        # Cost spikes
        cost_spikes = anomalies_data.get('cost_spikes', {}).get('anomalies', [])
        if cost_spikes:
            lines.append(f"\nCost Spikes: {len(cost_spikes)} detected\n")
            for spike in cost_spikes[:3]:
                lines.append(
                    f"- {spike['month']}: ${spike['cost']:,.2f} "
                    f"(deviation: {spike['deviation_pct']:+.1f}%, severity: {spike['severity']})\n"
                )
        
        # Service anomalies
        service_anomalies = anomalies_data.get('service_anomalies', {}).get('anomalies', [])
        if service_anomalies:
            lines.append(f"\nService Anomalies: {len(service_anomalies)} detected\n")
            for anomaly in service_anomalies[:3]:
                lines.append(f"- {anomaly.get('message', 'N/A')}\n")
        
        return "".join(lines)
    
    def _format_tagging_compliance(self, compliance_data: Dict) -> str:
        """Format tagging compliance data"""
        summary = compliance_data.get('summary', {})
        
        lines = [
            f"Compliance Rate: {summary.get('compliance_pct', 0):.1f}%\n",
            f"Untagged Resources: {summary.get('untagged_resources', 0)}\n",
            f"Cost of Untagged Resources: ${summary.get('untagged_cost', 0):,.2f} "
            f"({summary.get('untagged_cost_pct', 0):.1f}% of total)\n"
        ]
        
        return "".join(lines)
    
    def _format_kpis(self, kpis_data: Dict) -> str:
        """Format KPI data"""
        summary = kpis_data.get('summary', {})
        mom = kpis_data.get('month_over_month')
        
        lines = [
            f"Total Cost: ${summary.get('total_cost', 0):,.2f}\n",
            f"Record Count: {summary.get('record_count', 0)}\n",
        ]
        
        if mom:
            lines.append(
                f"Month-over-Month Change: {mom['percent_change']:+.2f}% "
                f"({mom['trend']})\n"
            )
        
        return "".join(lines)
    
    def calculate_recall_at_k(
        self,
        query: str,
        ground_truth_docs: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Recall@k for RAG evaluation
        
        Args:
            query: Query string
            ground_truth_docs: List of relevant document IDs/sources
            k: Number of documents retrieved
            
        Returns:
            float: Recall@k score (0.0 to 1.0)
        """
        logger.info(
            f"Calculating Recall@{k} for query: '{query[:50]}...'",
            extra={'request_id': self.request_id}
        )
        
        try:
            # Retrieve documents
            semantic_results = self._semantic_retrieval(query, k)
            retrieved_docs = semantic_results['sources']
            
            # Calculate recall
            if not ground_truth_docs:
                logger.warning(
                    "No ground truth documents provided",
                    extra={'request_id': self.request_id}
                )
                return 0.0
            
            # Count matches
            matches = len(set(retrieved_docs) & set(ground_truth_docs))
            recall = matches / len(ground_truth_docs)
            
            logger.info(
                f"Recall@{k} = {recall:.3f} ({matches}/{len(ground_truth_docs)} relevant docs retrieved)",
                extra={'request_id': self.request_id}
            )
            
            return recall
            
        except Exception as e:
            logger.error(
                f"Error calculating Recall@k: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            return 0.0


# Convenience function for initialization
def initialize_retriever(
    reference_docs_path: str = "data/reference",
    vector_store_path: str = "data/vector_store",
    database_url: str = "sqlite:///./data/cost_analytics.db",
    request_id: str = "init"
) -> Retriever:
    """
    Initialize complete retrieval system
    
    Args:
        reference_docs_path: Path to reference documents
        vector_store_path: Path to vector store
        database_url: Database connection string
        request_id: Request ID for logging
        
    Returns:
        Initialized Retriever instance
    """
    logger.info(
        "=" * 80,
        extra={'request_id': request_id}
    )
    logger.info(
        "INITIALIZING RETRIEVAL SYSTEM",
        extra={'request_id': request_id}
    )
    logger.info(
        "=" * 80,
        extra={'request_id': request_id}
    )
    
    try:
        # Initialize embeddings manager
        from .embeddings import initialize_embeddings
        
        embeddings_manager = initialize_embeddings(
            reference_docs_path=reference_docs_path,
            vector_store_path=vector_store_path,
            force_rebuild=False,
            request_id=request_id
        )
        
        # Create retriever
        retriever = Retriever(
            embeddings_manager=embeddings_manager,
            database_url=database_url,
            request_id=request_id
        )
        
        logger.info(
            "=" * 80,
            extra={'request_id': request_id}
        )
        logger.info(
            "✓ RETRIEVAL SYSTEM INITIALIZED",
            extra={'request_id': request_id}
        )
        logger.info(
            "=" * 80,
            extra={'request_id': request_id}
        )
        
        return retriever
        
    except Exception as e:
        logger.error(
            f"Failed to initialize retrieval system: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise
