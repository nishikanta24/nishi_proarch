"""
Data Transformation Module
Transforms raw billing data to required schema
"""

import pandas as pd
import logging
import json
from datetime import datetime
from typing import Tuple
from .parsers import parse_tags, extract_from_tags

logger = logging.getLogger(__name__)


def transform_to_schema(df: pd.DataFrame, request_id: str = "default") -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Transform raw FOCUS dataset to required billing and resource schema
    
    Args:
        df: Raw dataframe from CSV
        request_id: Request ID for logging
        
    Returns:
        tuple: (billing_df, resources_df, parse_errors_count, rows_skipped)
    """
    logger.info(
        "=" * 60,
        extra={'request_id': request_id}
    )
    logger.info(
        "STARTING DATA TRANSFORMATION",
        extra={'request_id': request_id}
    )
    logger.info(
        "=" * 60,
        extra={'request_id': request_id}
    )
    
    transform_start = datetime.now()
    
    # Validate required columns exist
    required_source_cols = [
        'BillingPeriodStart', 'BillingAccountId', 'SubAccountName',
        'ServiceName', 'ResourceId', 'RegionName', 'ConsumedQuantity',
        'ListUnitPrice', 'EffectiveCost', 'Tags'
    ]
    
    missing_cols = [col for col in required_source_cols if col not in df.columns]
    if missing_cols:
        logger.error(
            f"Missing required columns in source data: {missing_cols}",
            extra={'request_id': request_id}
        )
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(
        f"All required source columns present",
        extra={'request_id': request_id}
    )
    
    # Parse tags for all rows with progress tracking
    logger.info(
        f"Parsing tags for {len(df)} rows...",
        extra={'request_id': request_id}
    )
    
    parse_errors = 0
    parsed_tags_list = []
    
    for idx, row in df.iterrows():
        tags = parse_tags(row['Tags'], idx, request_id)
        parsed_tags_list.append(tags)
        # Count parse errors by checking if tags were expected but empty was returned
        if row['Tags'] and row['Tags'] != 'NULL' and row['Tags'] != '' and not pd.isna(row['Tags']) and not tags:
            parse_errors += 1
    
    df['parsed_tags'] = parsed_tags_list
    
    if parse_errors > 0:
        logger.warning(
            f"Encountered {parse_errors} tag parsing errors during transformation",
            extra={'request_id': request_id}
        )
    
    # Extract invoice_month from BillingPeriodStart
    logger.info(
        "Extracting invoice_month from BillingPeriodStart...",
        extra={'request_id': request_id}
    )
    
    try:
        df['invoice_month'] = pd.to_datetime(df['BillingPeriodStart']).dt.strftime('%Y-%m')
        unique_months = df['invoice_month'].unique()
        logger.info(
            f"Found {len(unique_months)} unique billing months: {sorted(unique_months)}",
            extra={'request_id': request_id}
        )
    except Exception as e:
        logger.error(
            f"Error parsing BillingPeriodStart dates: {str(e)}",
            extra={'request_id': request_id}
        )
        raise
    
    # Map columns to billing schema
    logger.info(
        "Mapping columns to billing schema...",
        extra={'request_id': request_id}
    )
    
    billing_df = pd.DataFrame({
        'invoice_month': df['invoice_month'],
        'account_id': df['BillingAccountId'].astype(str),
        'subscription': df['SubAccountName'],
        'service': df['ServiceName'],
        'resource_group': df['parsed_tags'].apply(lambda x: extract_from_tags(x, 'application')),
        'resource_id': df['ResourceId'],
        'region': df['RegionName'],
        'usage_qty': pd.to_numeric(df['ConsumedQuantity'], errors='coerce'),
        'unit_cost': pd.to_numeric(df['ListUnitPrice'], errors='coerce'),
        'cost': pd.to_numeric(df['EffectiveCost'], errors='coerce')
    })
    
    # Log transformation statistics
    logger.info(
        f"Billing records created: {len(billing_df)}",
        extra={'request_id': request_id}
    )
    logger.info(
        f"Resource group extraction: {billing_df['resource_group'].notna().sum()}/{len(billing_df)} records have resource_group",
        extra={'request_id': request_id}
    )
    
    # Create resources dataframe (unique resources)
    logger.info(
        "Creating unique resources dataset...",
        extra={'request_id': request_id}
    )
    
    resources_data = []
    seen_resources = set()
    skipped_null_resources = 0
    skipped_duplicate_resources = 0
    
    for idx, row in df.iterrows():
        resource_id = row['ResourceId']
        
        # Skip if null
        if pd.isna(resource_id) or resource_id == '' or resource_id == 'NULL':
            skipped_null_resources += 1
            continue
        
        # Skip if already processed
        if resource_id in seen_resources:
            skipped_duplicate_resources += 1
            continue
        
        seen_resources.add(resource_id)
        tags = row['parsed_tags']
        
        resources_data.append({
            'resource_id': resource_id,
            'owner': extract_from_tags(tags, 'owner'),
            'env': extract_from_tags(tags, 'environment'),
            'tags_json': json.dumps(tags) if tags else None
        })
    
    resources_df = pd.DataFrame(resources_data)
    
    rows_skipped = skipped_null_resources + skipped_duplicate_resources
    
    logger.info(
        f"Resources processed: {len(resources_df)} unique resources created",
        extra={'request_id': request_id}
    )
    logger.info(
        f"Skipped {skipped_null_resources} rows with null resource_id",
        extra={'request_id': request_id}
    )
    logger.info(
        f"Skipped {skipped_duplicate_resources} duplicate resource_ids",
        extra={'request_id': request_id}
    )
    
    # Log resource tag coverage
    owner_coverage = (resources_df['owner'].notna().sum() / len(resources_df) * 100) if len(resources_df) > 0 else 0
    env_coverage = (resources_df['env'].notna().sum() / len(resources_df) * 100) if len(resources_df) > 0 else 0
    
    logger.info(
        f"Resource tag coverage - Owner: {owner_coverage:.1f}%, Environment: {env_coverage:.1f}%",
        extra={'request_id': request_id}
    )
    
    transform_duration = (datetime.now() - transform_start).total_seconds()
    
    logger.info(
        "=" * 60,
        extra={'request_id': request_id}
    )
    logger.info(
        f"TRANSFORMATION COMPLETE in {transform_duration:.2f}s",
        extra={'request_id': request_id}
    )
    logger.info(
        f"Summary: {len(billing_df)} billing records, {len(resources_df)} unique resources",
        extra={'request_id': request_id}
    )
    logger.info(
        "=" * 60,
        extra={'request_id': request_id}
    )
    
    return billing_df, resources_df, parse_errors, rows_skipped