"""
Unit tests for Retriever functionality
Tests semantic search, structured data queries, and context assembly
"""

import pytest
import tempfile
import os
import sys
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from ai.retriever import Retriever, initialize_retriever
    from ai.embeddings import EmbeddingsManager
    from transformations.kpis import KPICalculator
except ImportError:
    # Fallback for pytest execution
    import sys
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from ai.retriever import Retriever, initialize_retriever
    from ai.embeddings import EmbeddingsManager
    from transformations.kpis import KPICalculator


class TestRetriever:
    """Test Retriever functionality"""

    def setup_method(self):
        """Setup for each test method"""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_url = f"sqlite:///{self.temp_db.name}"

        # Create temporary vector store path
        self.temp_vector_dir = tempfile.mkdtemp()
        self.vector_store_path = os.path.join(self.temp_vector_dir, 'vector_store')

        # Mock embeddings manager
        self.mock_embeddings = Mock(spec=EmbeddingsManager)

    def teardown_method(self):
        """Cleanup after each test method"""
        try:
            os.unlink(self.temp_db.name)
            import shutil
            shutil.rmtree(self.temp_vector_dir)
        except:
            pass

    @patch('ai.retriever.initialize_retriever')
    def test_retriever_initialization(self, mock_init):
        """Test retriever initialization"""
        mock_retriever = Mock()
        mock_init.return_value = mock_retriever

        # Test the initialization function
        retriever = initialize_retriever(
            reference_docs_path="data/reference",
            vector_store_path=self.vector_store_path,
            database_url=self.db_url,
            request_id="test-init"
        )

        mock_init.assert_called_once()
        assert retriever == mock_retriever

    def test_retriever_class_initialization(self):
        """Test Retriever class initialization with mocked dependencies"""
        with patch('ai.embeddings.EmbeddingsManager') as mock_embeddings_class, \
             patch('transformations.kpis.KPICalculator') as mock_kpi_class:

            mock_embeddings_instance = Mock()
            mock_kpi_instance = Mock()
            mock_embeddings_class.return_value = mock_embeddings_instance
            mock_kpi_class.return_value = mock_kpi_instance

            retriever = Retriever(
                embeddings_manager=mock_embeddings_instance,
                database_url=self.db_url,
                request_id="test-retriever"
            )

            assert retriever.embeddings_manager == mock_embeddings_instance
            assert retriever.kpi_calculator == mock_kpi_instance
            assert retriever.request_id == "test-retriever"

    @patch('ai.retriever.Retriever._semantic_retrieval')
    @patch('ai.retriever.Retriever._structured_retrieval')
    def test_retrieve_method_integration(self, mock_structured, mock_semantic):
        """Test the main retrieve method"""
        # Setup mocks
        mock_semantic.return_value = {
            'context': 'Semantic context from vector search',
            'sources': ['doc1.txt', 'doc2.txt'],
            'documents_retrieved': 2
        }

        mock_structured.return_value = {
            'context': 'Structured data from database',
            'data_points': 5
        }

        with patch('ai.embeddings.EmbeddingsManager') as mock_embeddings_class, \
             patch('transformations.kpis.KPICalculator') as mock_kpi_class:

            mock_embeddings_instance = Mock()
            mock_kpi_instance = Mock()
            mock_embeddings_class.return_value = mock_embeddings_instance
            mock_kpi_class.return_value = mock_kpi_instance

            retriever = Retriever(
                embeddings_manager=mock_embeddings_instance,
                database_url=self.db_url,
                request_id="test-retrieve"
            )

            # Test retrieve method
            result = retriever.retrieve("What is the total cost?", k=3)

            # Verify semantic retrieval was called
            mock_semantic.assert_called_once_with("What is the total cost?", 3)

            # Verify structured retrieval was called
            mock_structured.assert_called_once()

            # Check result structure
            assert 'context' in result
            assert 'sources' in result
            assert 'semantic_results' in result
            assert 'structured_results' in result
            assert 'retrieval_time_seconds' in result

    def test_context_assembly(self):
        """Test context assembly from multiple sources"""
        with patch('ai.embeddings.EmbeddingsManager') as mock_embeddings_class, \
             patch('transformations.kpis.KPICalculator') as mock_kpi_class:

            mock_embeddings_instance = Mock()
            mock_kpi_instance = Mock()
            mock_embeddings_class.return_value = mock_embeddings_instance
            mock_kpi_class.return_value = mock_kpi_instance

            retriever = Retriever(
                embeddings_manager=mock_embeddings_instance,
                database_url=self.db_url,
                request_id="test-context"
            )

            # Mock the retrieval methods
            with patch.object(retriever, '_semantic_retrieval') as mock_semantic, \
                 patch.object(retriever, '_structured_retrieval') as mock_structured:

                mock_semantic.return_value = {
                    'context': 'Reference documentation about cost optimization',
                    'sources': ['cost_optimization.txt'],
                    'documents_retrieved': 1
                }

                mock_structured.return_value = {
                    'context': 'Billing data shows $1000 total cost',
                    'data_points': 1
                }

                result = retriever.retrieve("What are my costs?", k=2)

                # Verify context contains both semantic and structured information
                assert 'cost optimization' in result['context'].lower()
                assert '$1000' in result['context']

                # Verify sources are tracked
                assert 'cost_optimization.txt' in result['sources']

    def test_semantic_retrieval_error_handling(self):
        """Test error handling in semantic retrieval"""
        with patch('ai.embeddings.EmbeddingsManager') as mock_embeddings_class, \
             patch('transformations.kpis.KPICalculator') as mock_kpi_class:

            mock_embeddings_instance = Mock()
            mock_kpi_instance = Mock()
            mock_embeddings_class.return_value = mock_embeddings_instance
            mock_kpi_class.return_value = mock_kpi_instance

            retriever = Retriever(
                embeddings_manager=mock_embeddings_instance,
                database_url=self.db_url,
                request_id="test-error"
            )

            # Mock semantic retrieval to raise exception
            with patch.object(retriever, '_semantic_retrieval') as mock_semantic:
                mock_semantic.side_effect = Exception("Vector search failed")

                with patch.object(retriever, '_structured_retrieval') as mock_structured:
                    mock_structured.return_value = {
                        'context': 'Fallback structured data',
                        'data_points': 1
                    }

                    # Should not raise exception, should fallback gracefully
                    result = retriever.retrieve("Test query", k=1)

                    assert 'context' in result
                    assert 'structured data' in result['context'].lower()

    def test_structured_retrieval_error_handling(self):
        """Test error handling in structured retrieval"""
        with patch('ai.embeddings.EmbeddingsManager') as mock_embeddings_class, \
             patch('transformations.kpis.KPICalculator') as mock_kpi_class:

            mock_embeddings_instance = Mock()
            mock_kpi_instance = Mock()
            mock_embeddings_class.return_value = mock_embeddings_instance
            mock_kpi_class.return_value = mock_kpi_instance

            retriever = Retriever(
                embeddings_manager=mock_embeddings_instance,
                database_url=self.db_url,
                request_id="test-structured-error"
            )

            # Mock structured retrieval to raise exception
            with patch.object(retriever, '_semantic_retrieval') as mock_semantic, \
                 patch.object(retriever, '_structured_retrieval') as mock_structured:

                mock_semantic.return_value = {
                    'context': 'Semantic context available',
                    'sources': ['doc1.txt'],
                    'documents_retrieved': 1
                }
                mock_structured.side_effect = Exception("Database query failed")

                # Should not raise exception, should continue with semantic only
                result = retriever.retrieve("Test query", k=1)

                assert 'context' in result
                assert 'semantic context' in result['context'].lower()

    @patch('ai.retriever.Retriever.calculate_recall_at_k')
    def test_recall_calculation_method(self, mock_recall):
        """Test recall calculation method exists and is callable"""
        mock_recall.return_value = 0.75

        with patch('ai.embeddings.EmbeddingsManager') as mock_embeddings_class, \
             patch('transformations.kpis.KPICalculator') as mock_kpi_class:

            mock_embeddings_instance = Mock()
            mock_kpi_instance = Mock()
            mock_embeddings_class.return_value = mock_embeddings_instance
            mock_kpi_class.return_value = mock_kpi_instance

            retriever = Retriever(
                embeddings_manager=mock_embeddings_instance,
                database_url=self.db_url,
                request_id="test-recall"
            )

            # Test recall calculation
            recall = retriever.calculate_recall_at_k("test query", ["expected_doc"], k=3)

            mock_recall.assert_called_once_with("test query", ["expected_doc"], 3)
            assert recall == 0.75

    def test_retrieval_performance_tracking(self):
        """Test that retrieval performance is tracked"""
        with patch('ai.embeddings.EmbeddingsManager') as mock_embeddings_class, \
             patch('transformations.kpis.KPICalculator') as mock_kpi_class:

            mock_embeddings_instance = Mock()
            mock_kpi_instance = Mock()
            mock_embeddings_class.return_value = mock_embeddings_instance
            mock_kpi_class.return_value = mock_kpi_instance

            retriever = Retriever(
                embeddings_manager=mock_embeddings_instance,
                database_url=self.db_url,
                request_id="test-performance"
            )

            with patch.object(retriever, '_semantic_retrieval') as mock_semantic, \
                 patch.object(retriever, '_structured_retrieval') as mock_structured:

                mock_semantic.return_value = {
                    'context': 'Semantic context',
                    'sources': ['doc1.txt'],
                    'documents_retrieved': 1
                }

                mock_structured.return_value = {
                    'context': 'Structured context',
                    'data_points': 2
                }

                result = retriever.retrieve("Performance test query", k=2)

                # Check that timing information is included
                assert 'retrieval_time_seconds' in result
                assert isinstance(result['retrieval_time_seconds'], (int, float))
                assert result['retrieval_time_seconds'] >= 0

    def test_empty_query_handling(self):
        """Test handling of empty or very short queries"""
        with patch('ai.embeddings.EmbeddingsManager') as mock_embeddings_class, \
             patch('transformations.kpis.KPICalculator') as mock_kpi_class:

            mock_embeddings_instance = Mock()
            mock_kpi_instance = Mock()
            mock_embeddings_class.return_value = mock_embeddings_instance
            mock_kpi_class.return_value = mock_kpi_instance

            retriever = Retriever(
                embeddings_manager=mock_embeddings_instance,
                database_url=self.db_url,
                request_id="test-empty"
            )

            with patch.object(retriever, '_semantic_retrieval') as mock_semantic, \
                 patch.object(retriever, '_structured_retrieval') as mock_structured:

                mock_semantic.return_value = {
                    'context': 'Default context for empty query',
                    'sources': ['default.txt'],
                    'documents_retrieved': 1
                }

                mock_structured.return_value = {
                    'context': 'Basic cost data',
                    'data_points': 1
                }

                # Test with empty query
                result = retriever.retrieve("", k=1)

                assert 'context' in result
                assert len(result['context']) > 0

                # Test with very short query
                result = retriever.retrieve("x", k=1)

                assert 'context' in result
                assert len(result['context']) > 0
