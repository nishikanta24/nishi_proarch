"""
Aggregations Module
Provides reusable aggregation functions for cost analysis
"""

import pandas as pd
import logging
from sqlalchemy import func, and_
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Optional, List

from ..ingestion.schema import Billing, Resource

logger = logging.getLogger(__name__)


def get_monthly_costs(engine, month: Optional[str] = None, request_id: str = "default") -> pd.DataFrame:
    """
    Get total costs by month
    
    Args:
        engine: SQLAlchemy engine
        month: Optional specific month in YYYY-MM format
        request_id: Request ID for logging
        
    Returns:
        pd.DataFrame: Monthly costs with columns [invoice_month, total_cost]
    """
    logger.info(
        f"Fetching monthly costs{f' for {month}' if month else ''}",
        extra={'request_id': request_id}
    )
    
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        
        query = session.query(
            Billing.invoice_month,
            func.sum(Billing.cost).label('total_cost'),
            func.count(Billing.id).label('record_count')
        )
        
        if month:
            query = query.filter(Billing.invoice_month == month)
        
        query = query.group_by(Billing.invoice_month).order_by(Billing.invoice_month)
        
        df = pd.read_sql(query.statement, engine)
        
        session.close()
        
        logger.info(
            f"Retrieved {len(df)} month(s) of cost data",
            extra={'request_id': request_id}
        )
        
        return df
        
    except Exception as e:
        logger.error(
            f"Error fetching monthly costs: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise


def get_costs_by_service(engine, month: str, request_id: str = "default") -> pd.DataFrame:
    """
    Get costs grouped by service for a specific month
    
    Args:
        engine: SQLAlchemy engine
        month: Month in YYYY-MM format
        request_id: Request ID for logging
        
    Returns:
        pd.DataFrame: Costs by service with columns [service, total_cost, record_count]
    """
    logger.info(
        f"Fetching costs by service for {month}",
        extra={'request_id': request_id}
    )
    
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        
        query = session.query(
            Billing.service,
            func.sum(Billing.cost).label('total_cost'),
            func.count(Billing.id).label('record_count'),
            func.avg(Billing.cost).label('avg_cost')
        ).filter(
            Billing.invoice_month == month
        ).group_by(
            Billing.service
        ).order_by(
            func.sum(Billing.cost).desc()
        )
        
        df = pd.read_sql(query.statement, engine)
        
        session.close()
        
        logger.info(
            f"Retrieved costs for {len(df)} services",
            extra={'request_id': request_id}
        )
        
        return df
        
    except Exception as e:
        logger.error(
            f"Error fetching costs by service: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise


def get_costs_by_resource_group(engine, month: str, request_id: str = "default") -> pd.DataFrame:
    """
    Get costs grouped by resource group for a specific month
    
    Args:
        engine: SQLAlchemy engine
        month: Month in YYYY-MM format
        request_id: Request ID for logging
        
    Returns:
        pd.DataFrame: Costs by resource group with columns [resource_group, total_cost, record_count]
    """
    logger.info(
        f"Fetching costs by resource group for {month}",
        extra={'request_id': request_id}
    )
    
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        
        query = session.query(
            Billing.resource_group,
            func.sum(Billing.cost).label('total_cost'),
            func.count(Billing.id).label('record_count'),
            func.avg(Billing.cost).label('avg_cost')
        ).filter(
            Billing.invoice_month == month
        ).group_by(
            Billing.resource_group
        ).order_by(
            func.sum(Billing.cost).desc()
        )
        
        df = pd.read_sql(query.statement, engine)
        
        session.close()
        
        logger.info(
            f"Retrieved costs for {len(df)} resource groups",
            extra={'request_id': request_id}
        )
        
        return df
        
    except Exception as e:
        logger.error(
            f"Error fetching costs by resource group: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise


def get_top_n_resources(engine, month: str, n: int = 10, request_id: str = "default") -> pd.DataFrame:
    """
    Get top N most expensive resources for a specific month
    
    Args:
        engine: SQLAlchemy engine
        month: Month in YYYY-MM format
        n: Number of top resources to return
        request_id: Request ID for logging
        
    Returns:
        pd.DataFrame: Top N resources with details
    """
    logger.info(
        f"Fetching top {n} resources for {month}",
        extra={'request_id': request_id}
    )
    
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Join with resources table to get metadata
        query = session.query(
            Billing.resource_id,
            Billing.service,
            Billing.resource_group,
            Billing.region,
            func.sum(Billing.cost).label('total_cost'),
            func.sum(Billing.usage_qty).label('total_usage'),
            Resource.owner,
            Resource.env
        ).outerjoin(
            Resource,
            Billing.resource_id == Resource.resource_id
        ).filter(
            Billing.invoice_month == month
        ).group_by(
            Billing.resource_id,
            Billing.service,
            Billing.resource_group,
            Billing.region,
            Resource.owner,
            Resource.env
        ).order_by(
            func.sum(Billing.cost).desc()
        ).limit(n)
        
        df = pd.read_sql(query.statement, engine)
        
        session.close()
        
        logger.info(
            f"Retrieved top {len(df)} resources",
            extra={'request_id': request_id}
        )
        
        return df
        
    except Exception as e:
        logger.error(
            f"Error fetching top N resources: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise


def get_monthly_trend(engine, months: int = 6, request_id: str = "default") -> pd.DataFrame:
    """
    Get cost trend for last N months
    
    Args:
        engine: SQLAlchemy engine
        months: Number of months to retrieve
        request_id: Request ID for logging
        
    Returns:
        pd.DataFrame: Monthly trend with growth rates
    """
    logger.info(
        f"Fetching {months}-month cost trend",
        extra={'request_id': request_id}
    )
    
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        
        query = session.query(
            Billing.invoice_month,
            func.sum(Billing.cost).label('total_cost'),
            func.count(Billing.id).label('record_count')
        ).group_by(
            Billing.invoice_month
        ).order_by(
            Billing.invoice_month.desc()
        ).limit(months)
        
        df = pd.read_sql(query.statement, engine)
        
        # Reverse to chronological order
        df = df.sort_values('invoice_month').reset_index(drop=True)
        
        # Calculate month-over-month growth
        if len(df) > 1:
            df['prev_month_cost'] = df['total_cost'].shift(1)
            df['mom_change'] = df['total_cost'] - df['prev_month_cost']
            df['mom_change_pct'] = (df['mom_change'] / df['prev_month_cost'] * 100).round(2)
        
        session.close()
        
        logger.info(
            f"Retrieved {len(df)} months of trend data",
            extra={'request_id': request_id}
        )
        
        return df
        
    except Exception as e:
        logger.error(
            f"Error fetching monthly trend: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise


def get_cost_breakdown(engine, month: str, group_by: str, request_id: str = "default") -> pd.DataFrame:
    """
    Flexible cost breakdown by different dimensions
    
    Args:
        engine: SQLAlchemy engine
        month: Month in YYYY-MM format
        group_by: Dimension to group by (service, region, account_id, resource_group)
        request_id: Request ID for logging
        
    Returns:
        pd.DataFrame: Cost breakdown by specified dimension
    """
    logger.info(
        f"Fetching cost breakdown by {group_by} for {month}",
        extra={'request_id': request_id}
    )
    
    valid_dimensions = ['service', 'region', 'account_id', 'resource_group', 'subscription']
    
    if group_by not in valid_dimensions:
        raise ValueError(f"Invalid group_by dimension. Must be one of: {valid_dimensions}")
    
    try:
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Dynamically get the column to group by
        group_column = getattr(Billing, group_by)
        
        query = session.query(
            group_column.label('dimension'),
            func.sum(Billing.cost).label('total_cost'),
            func.count(Billing.id).label('record_count'),
            func.avg(Billing.cost).label('avg_cost'),
            func.min(Billing.cost).label('min_cost'),
            func.max(Billing.cost).label('max_cost')
        ).filter(
            Billing.invoice_month == month
        ).group_by(
            group_column
        ).order_by(
            func.sum(Billing.cost).desc()
        )
        
        df = pd.read_sql(query.statement, engine)
        
        # Calculate percentage of total
        total_cost = df['total_cost'].sum()
        df['pct_of_total'] = (df['total_cost'] / total_cost * 100).round(2)
        
        session.close()
        
        logger.info(
            f"Retrieved breakdown across {len(df)} {group_by} values",
            extra={'request_id': request_id}
        )
        
        return df
        
    except Exception as e:
        logger.error(
            f"Error fetching cost breakdown: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise


def get_unit_cost_changes(engine, current_month: str, previous_month: str, 
                          request_id: str = "default") -> pd.DataFrame:
    """
    Detect unit cost changes between two months for the same resources
    
    Args:
        engine: SQLAlchemy engine
        current_month: Current month in YYYY-MM format
        previous_month: Previous month in YYYY-MM format
        request_id: Request ID for logging
        
    Returns:
        pd.DataFrame: Resources with unit cost changes
    """
    logger.info(
        f"Detecting unit cost changes between {previous_month} and {current_month}",
        extra={'request_id': request_id}
    )
    
    try:
        # Get current month data
        query_current = f"""
        SELECT 
            resource_id,
            service,
            resource_group,
            AVG(unit_cost) as current_unit_cost,
            SUM(usage_qty) as current_usage,
            SUM(cost) as current_cost
        FROM billing
        WHERE invoice_month = '{current_month}'
        GROUP BY resource_id, service, resource_group
        """
        
        df_current = pd.read_sql(query_current, engine)
        
        # Get previous month data
        query_previous = f"""
        SELECT 
            resource_id,
            service,
            resource_group,
            AVG(unit_cost) as previous_unit_cost,
            SUM(usage_qty) as previous_usage,
            SUM(cost) as previous_cost
        FROM billing
        WHERE invoice_month = '{previous_month}'
        GROUP BY resource_id, service, resource_group
        """
        
        df_previous = pd.read_sql(query_previous, engine)
        
        # Merge to compare
        df_merged = df_current.merge(
            df_previous,
            on=['resource_id', 'service', 'resource_group'],
            how='inner'
        )
        
        # Calculate changes
        df_merged['unit_cost_change'] = df_merged['current_unit_cost'] - df_merged['previous_unit_cost']
        df_merged['unit_cost_change_pct'] = (
            df_merged['unit_cost_change'] / df_merged['previous_unit_cost'] * 100
        ).round(2)
        
        # Filter significant changes (>5%)
        df_significant = df_merged[abs(df_merged['unit_cost_change_pct']) > 5].copy()
        df_significant = df_significant.sort_values('unit_cost_change_pct', ascending=False)
        
        logger.info(
            f"Found {len(df_significant)} resources with significant unit cost changes (>5%)",
            extra={'request_id': request_id}
        )
        
        return df_significant
        
    except Exception as e:
        logger.error(
            f"Error detecting unit cost changes: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise


def get_resources_without_tags(engine, month: str, request_id: str = "default") -> pd.DataFrame:
    """
    Get resources with missing tags (owner or environment)
    
    Args:
        engine: SQLAlchemy engine
        month: Month in YYYY-MM format
        request_id: Request ID for logging
        
    Returns:
        pd.DataFrame: Resources with missing tags and their costs
    """
    logger.info(
        f"Fetching resources with missing tags for {month}",
        extra={'request_id': request_id}
    )
    
    try:
        query = f"""
        SELECT 
            b.resource_id,
            b.service,
            b.resource_group,
            b.region,
            SUM(b.cost) as total_cost,
            r.owner,
            r.env,
            CASE 
                WHEN r.owner IS NULL AND r.env IS NULL THEN 'both'
                WHEN r.owner IS NULL THEN 'owner'
                WHEN r.env IS NULL THEN 'environment'
            END as missing_tags
        FROM billing b
        LEFT JOIN resources r ON b.resource_id = r.resource_id
        WHERE b.invoice_month = '{month}'
        AND (r.owner IS NULL OR r.env IS NULL)
        GROUP BY b.resource_id, b.service, b.resource_group, b.region, r.owner, r.env
        ORDER BY total_cost DESC
        """
        
        df = pd.read_sql(query, engine)
        
        logger.info(
            f"Found {len(df)} resources with missing tags",
            extra={'request_id': request_id}
        )
        
        return df
        
    except Exception as e:
        logger.error(
            f"Error fetching resources without tags: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise