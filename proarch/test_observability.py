#!/usr/bin/env python3
"""
Test script to verify token usage tracking and cache metrics functionality
"""

import os
import sys
import time
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_token_usage_tracking():
    """Test that token usage is tracked in QA responses"""
    print("🔬 Testing Token Usage Tracking...")

    try:
        from ai.qa_chain import QACopilot

        # Create QA copilot with mock LLM
        with patch('ai.qa_chain.ChatOpenAI') as mock_llm_class:
            mock_llm = Mock()
            mock_llm_class.return_value = mock_llm

            # Mock the callback to return token usage
            mock_callback = Mock()
            mock_callback.total_tokens = 150
            mock_callback.prompt_tokens = 100
            mock_callback.completion_tokens = 50

            qa_copilot = QACopilot(request_id="test-token")

            # Test token tracking method
            token_metrics = qa_copilot._track_token_usage(mock_callback)

            assert 'total_tokens' in token_metrics
            assert 'prompt_tokens' in token_metrics
            assert 'completion_tokens' in token_metrics
            assert 'estimated_cost_usd' in token_metrics

            assert token_metrics['total_tokens'] == 150
            assert token_metrics['prompt_tokens'] == 100
            assert token_metrics['completion_tokens'] == 50
            assert token_metrics['estimated_cost_usd'] > 0

            print("✅ Token usage tracking works correctly")
            print(f"   Sample metrics: {token_metrics}")
            return True

    except Exception as e:
        print(f"❌ Token usage tracking test failed: {e}")
        return False

def test_cache_functionality():
    """Test that cache metrics are tracked in retriever"""
    print("\n🔬 Testing Cache Functionality...")

    try:
        from ai.retriever import Retriever
        from ai.embeddings import EmbeddingsManager

        # Create mock components
        with patch('ai.embeddings.EmbeddingsManager') as mock_embeddings_class, \
             patch('transformations.kpis.KPICalculator') as mock_kpi_class, \
             patch('transformations.anomalies.AnomalyDetector') as mock_anomaly_class:

            mock_embeddings = Mock()
            mock_kpi = Mock()
            mock_anomaly = Mock()

            mock_embeddings_class.return_value = mock_embeddings
            mock_kpi_class.return_value = mock_kpi
            mock_anomaly_class.return_value = mock_anomaly

            # Set environment variable for cache
            os.environ['ENABLE_CACHE'] = 'true'
            os.environ['CACHE_TTL_MINUTES'] = '5'

            retriever = Retriever(
                embeddings_manager=mock_embeddings,
                database_url="sqlite:///./data/test.db",
                request_id="test-cache"
            )

            # Test cache metrics
            metrics = retriever.get_cache_metrics()

            assert 'cache_enabled' in metrics
            assert 'cache_hits' in metrics
            assert 'cache_misses' in metrics
            assert 'cache_hit_rate_percent' in metrics

            assert metrics['cache_enabled'] == True
            assert metrics['cache_hits'] == 0
            assert metrics['cache_misses'] == 0

            print("✅ Cache functionality works correctly")
            print(f"   Initial metrics: {metrics}")

            # Test cache key generation
            key1 = retriever._get_cache_key("test question", 5, True)
            key2 = retriever._get_cache_key("test question", 5, True)
            key3 = retriever._get_cache_key("different question", 5, True)

            assert key1 == key2  # Same inputs should generate same key
            assert key1 != key3  # Different inputs should generate different key

            print("✅ Cache key generation works correctly")
            print(f"   Key1: {key1[:16]}...")
            print(f"   Key3: {key3[:16]}...")

            # Clean up environment
            del os.environ['ENABLE_CACHE']
            del os.environ['CACHE_TTL_MINUTES']

            return True

    except Exception as e:
        print(f"❌ Cache functionality test failed: {e}")
        return False

def test_cost_estimation():
    """Test that cost estimation works correctly"""
    print("\n🔬 Testing Cost Estimation...")

    try:
        from ai.qa_chain import QACopilot

        qa_copilot = QACopilot(request_id="test-cost")

        # Test cost calculation
        cost = qa_copilot._estimate_cost(1000, 500)  # 1000 prompt, 500 completion tokens

        # Expected cost calculation:
        # Prompt: 1000 tokens / 1000 * $0.0014 = $0.0014
        # Completion: 500 tokens / 1000 * $0.0028 = $0.0014
        # Total: $0.0028
        expected_cost = 0.0028

        assert abs(cost - expected_cost) < 0.0001  # Allow small floating point differences

        print("✅ Cost estimation works correctly")
        print(f"   Cost for 1000 prompt + 500 completion tokens: ${cost:.6f}")

        return True

    except Exception as e:
        print(f"❌ Cost estimation test failed: {e}")
        return False

def main():
    """Run all observability tests"""
    print("🚀 OBSERVABILITY ENHANCEMENT TESTS")
    print("=" * 50)

    results = []

    # Test token usage tracking
    results.append(("Token Usage Tracking", test_token_usage_tracking()))

    # Test cache functionality
    results.append(("Cache Functionality", test_cache_functionality()))

    # Test cost estimation
    results.append(("Cost Estimation", test_cost_estimation()))

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1

    success_rate = (passed / total) * 100

    print(f"\n🏆 OVERALL RESULT: {passed}/{total} tests passed ({success_rate:.1f}%)")

    if success_rate == 100.0:
        print("🎉 All observability enhancements are working correctly!")
        print("\n✨ New Features Available:")
        print("   • Token usage tracking in /ask responses")
        print("   • Cost estimation for LLM calls")
        print("   • Retrieval caching with configurable TTL")
        print("   • Cache hit/miss metrics in all responses")
    else:
        print("⚠️  Some tests failed. Check the implementation.")

    return success_rate == 100.0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
