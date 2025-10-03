"""
Data Quality Checks Module
Runs comprehensive quality validation on transformed billing data
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Custom exception for data quality issues"""
    pass


def run_quality_checks(billing_df: pd.DataFrame, resources_df: pd.DataFrame, request_id: str = "default") -> Dict:
    """
    Run comprehensive data quality checks on transformed data
    
    Checks:
    1. Null values in critical fields
    2. Negative costs
    3. Duplicate resource IDs
    4. Date format validation
    5. Missing tags coverage
    
    Args:
        billing_df: Billing dataframe
        resources_df: Resources dataframe
        request_id: Request ID for logging
        
    Returns:
        dict: Quality check results with detailed metrics
    """
    logger.info(
        "=" * 60,
        extra={'request_id': request_id}
    )
    logger.info(
        "STARTING QUALITY CHECKS",
        extra={'request_id': request_id}
    )
    logger.info(
        "=" * 60,
        extra={'request_id': request_id}
    )
    
    check_start = datetime.now()
    
    results = {
        'passed': True,
        'checks': [],
        'summary': {}
    }
    
    quality_check_failures = 0
    
    # Check 1: Null values in critical fields
    logger.info(
        "Check 1/5: Validating critical fields for null values...",
        extra={'request_id': request_id}
    )
    
    critical_fields = ['invoice_month', 'account_id', 'service', 'cost']
    null_counts = billing_df[critical_fields].isnull().sum()
    
    for field, count in null_counts.items():
        check = {
            'name': f'null_check_{field}',
            'passed': count == 0,
            'message': f"Found {count} null values in {field}" if count > 0 else f"No null values in {field}",
            'severity': 'error' if count > 0 else 'info',
            'affected_rows': int(count)
        }
        results['checks'].append(check)
        
        if not check['passed']:
            results['passed'] = False
            quality_check_failures += 1
            logger.error(
                f"❌ {check['message']}",
                extra={'request_id': request_id}
            )
        else:
            logger.info(
                f"✓ {check['message']}",
                extra={'request_id': request_id}
            )
    
    # Check 2: Negative costs
    logger.info(
        "Check 2/5: Checking for negative costs...",
        extra={'request_id': request_id}
    )
    
    negative_costs = (billing_df['cost'] < 0).sum()
    negative_cost_total = billing_df[billing_df['cost'] < 0]['cost'].sum() if negative_costs > 0 else 0
    
    check = {
        'name': 'negative_cost_check',
        'passed': negative_costs == 0,
        'message': f"Found {negative_costs} records with negative costs (total: ${negative_cost_total:.2f})" if negative_costs > 0 else "No negative costs found",
        'severity': 'warning' if negative_costs > 0 else 'info',
        'affected_rows': int(negative_costs)
    }
    results['checks'].append(check)
    
    if not check['passed']:
        logger.warning(
            f"⚠️  {check['message']}",
            extra={'request_id': request_id}
        )
    else:
        logger.info(
            f"✓ {check['message']}",
            extra={'request_id': request_id}
        )
    
    # Check 3: Duplicate resource IDs
    logger.info(
        "Check 3/5: Checking for duplicate resource IDs...",
        extra={'request_id': request_id}
    )
    
    duplicate_resources = resources_df['resource_id'].duplicated().sum()
    
    check = {
        'name': 'duplicate_resource_check',
        'passed': duplicate_resources == 0,
        'message': f"Found {duplicate_resources} duplicate resource IDs" if duplicate_resources > 0 else "No duplicate resource IDs found",
        'severity': 'error' if duplicate_resources > 0 else 'info',
        'affected_rows': int(duplicate_resources)
    }
    results['checks'].append(check)
    
    if not check['passed']:
        results['passed'] = False
        quality_check_failures += 1
        logger.error(
            f"❌ {check['message']}",
            extra={'request_id': request_id}
        )
    else:
        logger.info(
            f"✓ {check['message']}",
            extra={'request_id': request_id}
        )
    
    # Check 4: Date format validation
    logger.info(
        "Check 4/5: Validating date formats...",
        extra={'request_id': request_id}
    )
    
    try:
        pd.to_datetime(billing_df['invoice_month'], format='%Y-%m')
        date_check_passed = True
        date_message = f"All {len(billing_df)} dates are in valid YYYY-MM format"
    except Exception as e:
        date_check_passed = False
        date_message = f"Invalid date format found: {str(e)}"
    
    check = {
        'name': 'date_format_check',
        'passed': date_check_passed,
        'message': date_message,
        'severity': 'error' if not date_check_passed else 'info',
        'affected_rows': 0 if date_check_passed else len(billing_df)
    }
    results['checks'].append(check)
    
    if not check['passed']:
        results['passed'] = False
        quality_check_failures += 1
        logger.error(
            f"❌ {check['message']}",
            extra={'request_id': request_id}
        )
    else:
        logger.info(
            f"✓ {check['message']}",
            extra={'request_id': request_id}
        )
    
    # Check 5: Missing tags coverage
    logger.info(
        "Check 5/5: Analyzing tag coverage...",
        extra={'request_id': request_id}
    )
    
    missing_owner = resources_df['owner'].isnull().sum()
    missing_env = resources_df['env'].isnull().sum()
    missing_both = ((resources_df['owner'].isnull()) & (resources_df['env'].isnull())).sum()
    total_resources = len(resources_df)
    
    tag_coverage = {
        'owner_coverage_pct': ((total_resources - missing_owner) / total_resources * 100) if total_resources > 0 else 0,
        'env_coverage_pct': ((total_resources - missing_env) / total_resources * 100) if total_resources > 0 else 0,
        'missing_owner': int(missing_owner),
        'missing_env': int(missing_env),
        'missing_both': int(missing_both)
    }
    
    check = {
        'name': 'tag_coverage_check',
        'passed': True,  # This is informational
        'message': f"Tag coverage - Owner: {tag_coverage['owner_coverage_pct']:.1f}%, Environment: {tag_coverage['env_coverage_pct']:.1f}%",
        'severity': 'info',
        'details': tag_coverage,
        'affected_rows': 0
    }
    results['checks'].append(check)
    
    logger.info(
        f"ℹ️  {check['message']}",
        extra={'request_id': request_id}
    )
    logger.info(
        f"   - Resources missing owner tag: {missing_owner}",
        extra={'request_id': request_id}
    )
    logger.info(
        f"   - Resources missing environment tag: {missing_env}",
        extra={'request_id': request_id}
    )
    logger.info(
        f"   - Resources missing both tags: {missing_both}",
        extra={'request_id': request_id}
    )
    
    # Calculate summary
    passed_checks = sum(1 for c in results['checks'] if c['passed'])
    total_checks = len(results['checks'])
    error_checks = [c for c in results['checks'] if c['severity'] == 'error' and not c['passed']]
    warning_checks = [c for c in results['checks'] if c['severity'] == 'warning' and not c['passed']]
    
    results['summary'] = {
        'total_checks': total_checks,
        'passed_checks': passed_checks,
        'failed_checks': total_checks - passed_checks,
        'error_count': len(error_checks),
        'warning_count': len(warning_checks),
        'quality_check_failures': quality_check_failures
    }
    
    check_duration = (datetime.now() - check_start).total_seconds()
    
    logger.info(
        "=" * 60,
        extra={'request_id': request_id}
    )
    logger.info(
        f"QUALITY CHECKS COMPLETE in {check_duration:.2f}s",
        extra={'request_id': request_id}
    )
    logger.info(
        f"Results: {passed_checks}/{total_checks} checks passed",
        extra={'request_id': request_id}
    )
    
    if error_checks:
        logger.error(
            f"Found {len(error_checks)} critical errors that must be fixed",
            extra={'request_id': request_id}
        )
    if warning_checks:
        logger.warning(
            f"Found {len(warning_checks)} warnings that should be reviewed",
            extra={'request_id': request_id}
        )
    
    logger.info(
        "=" * 60,
        extra={'request_id': request_id}
    )
    
    return results