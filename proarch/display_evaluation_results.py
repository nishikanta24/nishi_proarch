#!/usr/bin/env python3
"""
Display RAG Evaluation Results
Shows the evaluation summary in a readable format
"""

import json
import statistics

def main():
    try:
        with open('rag_evaluation_report.json', 'r') as f:
            report = json.load(f)

        print("=" * 80)
        print("RAG EVALUATION SUMMARY")
        print("=" * 80)

        # Overall metrics
        overall = report['evaluation_summary']['overall_metrics']
        print(f"Total Test Cases: {overall['total_test_cases']}")
        print(f"Successful Evaluations: {overall['successful_evaluations']}")
        print(f"Success Rate: {overall['success_rate_percent']:.1f}%")
        print(f"Mean Execution Time: {overall['mean_execution_time']:.2f}s")
        print()

        # Retrieval metrics
        print("RETRIEVAL METRICS:")
        retrieval = report['evaluation_summary']['retrieval_metrics']
        for k in [1, 3, 5]:
            mean_recall = retrieval.get(f'mean_recall_at_{k}', 0)
            std_recall = retrieval.get(f'std_recall_at_{k}', 0)
            print(f"  Recall@{k}: {mean_recall:.3f} +/- {std_recall:.3f}")
        print()

        # Answer quality metrics
        print("ANSWER QUALITY METRICS:")
        quality = report['evaluation_summary']['answer_quality_scores']
        for criterion in ['accuracy', 'completeness', 'clarity', 'actionability', 'overall_score']:
            mean_score = quality.get(f'mean_{criterion}', 0)
            std_score = quality.get(f'std_{criterion}', 0)
            print(f"  {criterion.title()}: {mean_score:.2f} +/- {std_score:.2f}")
        print()

        # Recommendations
        recommendations = report['evaluation_summary'].get('recommendations', [])
        if recommendations:
            print("RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("RECOMMENDATIONS: None")

        print("=" * 80)

        # Summary interpretation
        print("\nINTERPRETATION:")
        success_rate = overall['success_rate_percent']
        mean_accuracy = quality.get('mean_accuracy', 0)
        mean_completeness = quality.get('mean_completeness', 0)
        mean_overall = quality.get('mean_overall_score', 0)

        if success_rate == 100.0:
            print("[PASS] All test cases completed successfully")
        else:
            print(f"[WARN] {100-success_rate:.1f}% of test cases failed")

        if mean_accuracy >= 4.5:
            print("[EXCELLENT] Answer accuracy")
        elif mean_accuracy >= 4.0:
            print("[GOOD] Answer accuracy")
        else:
            print("[NEEDS IMPROVEMENT] Answer accuracy")

        if mean_completeness >= 4.5:
            print("[EXCELLENT] Answer completeness")
        elif mean_completeness >= 4.0:
            print("[GOOD] Answer completeness")
        else:
            print("[NEEDS IMPROVEMENT] Answer completeness")

        if mean_overall >= 4.0:
            print("[EXCELLENT] Overall answer quality")
        elif mean_overall >= 3.0:
            print("[GOOD] Overall answer quality")
        else:
            print("[NEEDS IMPROVEMENT] Overall answer quality")

        print("=" * 80)

    except FileNotFoundError:
        print("Error: rag_evaluation_report.json not found. Run evaluation first.")
    except Exception as e:
        print(f"Error reading evaluation report: {str(e)}")

if __name__ == "__main__":
    main()
