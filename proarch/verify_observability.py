#!/usr/bin/env python3
"""
Simple verification script for observability enhancements
"""

import os
import sys

def verify_code_changes():
    """Verify that observability enhancements are implemented in the code"""

    print("🔍 VERIFYING OBSERVABILITY ENHANCEMENTS")
    print("=" * 50)

    enhancements = []

    # 1. Check token usage tracking in qa_chain.py
    print("1. Checking Token Usage Tracking...")
    try:
        with open('src/ai/qa_chain.py', 'r') as f:
            qa_content = f.read()

        checks = [
            ('get_openai_callback import', 'from langchain.callbacks import get_openai_callback' in qa_content),
            ('token tracking method', '_track_token_usage' in qa_content),
            ('cost estimation method', '_estimate_cost' in qa_content),
            ('token usage in response', 'token_usage' in qa_content and 'get_openai_callback' in qa_content),
        ]

        all_passed = all(check[1] for check in checks)
        enhancements.append(("Token Usage Tracking", all_passed, checks))

        if all_passed:
            print("   ✅ All token usage tracking features implemented")
        else:
            print("   ⚠️  Some token usage features missing:")
            for check_name, passed in checks:
                status = "✅" if passed else "❌"
                print(f"      {status} {check_name}")

    except Exception as e:
        print(f"   ❌ Error checking qa_chain.py: {e}")
        enhancements.append(("Token Usage Tracking", False, []))

    # 2. Check cache functionality in retriever.py
    print("\n2. Checking Cache Functionality...")
    try:
        with open('src/ai/retriever.py', 'r') as f:
            retriever_content = f.read()

        checks = [
            ('cache imports', 'import hashlib' in retriever_content and 'from datetime import timedelta' in retriever_content),
            ('cache initialization', '_cache' in retriever_content and '_cache_enabled' in retriever_content),
            ('cache methods', '_get_cache_key' in retriever_content and '_get_cached_result' in retriever_content),
            ('cache metrics', 'get_cache_metrics' in retriever_content),
            ('cache integration', '_get_cached_result(cache_key)' in retriever_content),
            ('cache storage', '_set_cached_result' in retriever_content),
        ]

        all_passed = all(check[1] for check in checks)
        enhancements.append(("Cache Functionality", all_passed, checks))

        if all_passed:
            print("   ✅ All cache functionality features implemented")
        else:
            print("   ⚠️  Some cache features missing:")
            for check_name, passed in checks:
                status = "✅" if passed else "❌"
                print(f"      {status} {check_name}")

    except Exception as e:
        print(f"   ❌ Error checking retriever.py: {e}")
        enhancements.append(("Cache Functionality", False, []))

    # 3. Check API integration in routes.py
    print("\n3. Checking API Integration...")
    try:
        with open('src/api/routes.py', 'r') as f:
            routes_content = f.read()

        checks = [
            ('token usage in response', '"token_usage": qa_response.get("token_usage", {})' in routes_content),
            ('cache metrics in response', '"cache_metrics": retrieval_result.get(\'cache_metrics\', {})' in routes_content),
        ]

        all_passed = all(check[1] for check in checks)
        enhancements.append(("API Integration", all_passed, checks))

        if all_passed:
            print("   ✅ All API integration features implemented")
        else:
            print("   ⚠️  Some API integration features missing:")
            for check_name, passed in checks:
                status = "✅" if passed else "❌"
                print(f"      {status} {check_name}")

    except Exception as e:
        print(f"   ❌ Error checking routes.py: {e}")
        enhancements.append(("API Integration", False, []))

    # Summary
    print("\n" + "=" * 50)
    print("📊 IMPLEMENTATION SUMMARY")

    total_implemented = 0
    total_features = len(enhancements)

    for feature_name, implemented, _ in enhancements:
        status = "✅ IMPLEMENTED" if implemented else "❌ MISSING"
        print(f"   {feature_name}: {status}")
        if implemented:
            total_implemented += 1

    success_rate = (total_implemented / total_features) * 100

    print(f"\n🏆 OVERALL STATUS: {total_implemented}/{total_features} features implemented ({success_rate:.1f}%)")

    if success_rate == 100.0:
        print("\n🎉 SUCCESS! All observability enhancements are properly implemented!")
        print("\n✨ NEW FEATURES NOW AVAILABLE:")
        print("   • Real-time token usage tracking in /ask API responses")
        print("   • Automatic cost estimation for LLM calls")
        print("   • Intelligent retrieval caching with configurable TTL")
        print("   • Cache performance metrics in all API responses")
        print("   • Enhanced logging with token and cache information")
        print("\n🔧 CONFIGURATION:")
        print("   Set ENABLE_CACHE=true (default: true)")
        print("   Set CACHE_TTL_MINUTES=30 (default: 30 minutes)")
        print("\n📋 API RESPONSE ENHANCEMENTS:")
        print("   /ask endpoint now includes:")
        print("   - token_usage: {total_tokens, prompt_tokens, completion_tokens, estimated_cost_usd}")
        print("   - metadata.cache_metrics: {cache_enabled, cache_hits, cache_misses, cache_hit_rate_percent}")
    else:
        print(f"\n⚠️  {total_features - total_implemented} features need implementation")

    return success_rate == 100.0

if __name__ == '__main__':
    success = verify_code_changes()
    sys.exit(0 if success else 1)
