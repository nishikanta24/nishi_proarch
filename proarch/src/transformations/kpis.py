"""
KPIs Module
Calculates and formats key performance indicators for cost analytics
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from .aggregations import (
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


class KPICalculator:
    """Calculates KPIs for cost analytics"""
    
    def __init__(self, engine, request_id: str = "default"):
        self.engine = engine
        self.request_id = request_id
        
        logger.info(
            "KPICalculator initialized",
            extra={'request_id': request_id}
        )
    
    def calculate_monthly_kpis(self, month: str) -> Dict:
        """
        Calculate comprehensive KPIs for a specific month
        
        Args:
            month: Month in YYYY-MM format
            
        Returns:
            dict: Complete KPI package with all metrics
        """
        logger.info(
            f"Calculating KPIs for month: {month}",
            extra={'request_id': self.request_id}
        )
        
        kpi_start = datetime.now()
        
        try:
            # Get previous month for comparisons
            month_dt = datetime.strptime(month, '%Y-%m')
            prev_month_dt = month_dt - relativedelta(months=1)
            prev_month = prev_month_dt.strftime('%Y-%m')
            
            logger.info(
                f"Comparing {month} with previous month {prev_month}",
                extra={'request_id': self.request_id}
            )
            
            # 1. Total monthly cost
            monthly_costs = get_monthly_costs(self.engine, month, self.request_id)
            
            if monthly_costs.empty:
                logger.warning(
                    f"No data found for month {month}",
                    extra={'request_id': self.request_id}
                )
                return {
                    'status': 'no_data',
                    'month': month,
                    'message': f'No billing data found for {month}'
                }
            
            total_cost = float(monthly_costs['total_cost'].iloc[0])
            record_count = int(monthly_costs['record_count'].iloc[0])
            
            logger.info(
                f"Total cost for {month}: ${total_cost:,.2f} ({record_count} records)",
                extra={'request_id': self.request_id}
            )
            
            # 2. Previous month comparison
            prev_monthly_costs = get_monthly_costs(self.engine, prev_month, self.request_id)
            
            month_over_month = None
            if not prev_monthly_costs.empty:
                prev_total_cost = float(prev_monthly_costs['total_cost'].iloc[0])
                cost_change = total_cost - prev_total_cost
                cost_change_pct = (cost_change / prev_total_cost * 100) if prev_total_cost > 0 else 0
                
                month_over_month = {
                    'previous_month': prev_month,
                    'previous_cost': round(prev_total_cost, 2),
                    'current_cost': round(total_cost, 2),
                    'absolute_change': round(cost_change, 2),
                    'percent_change': round(cost_change_pct, 2),
                    'trend': 'up' if cost_change > 0 else 'down' if cost_change < 0 else 'flat'
                }
                
                logger.info(
                    f"MoM change: {cost_change_pct:+.2f}% ({month_over_month['trend']})",
                    extra={'request_id': self.request_id}
                )
            
            # 3. Costs by service
            service_costs = get_costs_by_service(self.engine, month, self.request_id)
            
            top_services = []
            if not service_costs.empty:
                service_costs['pct_of_total'] = (service_costs['total_cost'] / total_cost * 100).round(2)
                
                for _, row in service_costs.head(10).iterrows():
                    top_services.append({
                        'service': row['service'],
                        'cost': round(float(row['total_cost']), 2),
                        'pct_of_total': float(row['pct_of_total']),
                        'record_count': int(row['record_count']),
                        'avg_cost': round(float(row['avg_cost']), 2)
                    })
                
                logger.info(
                    f"Top service: {top_services[0]['service']} (${top_services[0]['cost']:,.2f}, {top_services[0]['pct_of_total']}%)",
                    extra={'request_id': self.request_id}
                )
            
            # 4. Costs by resource group
            rg_costs = get_costs_by_resource_group(self.engine, month, self.request_id)
            
            top_resource_groups = []
            if not rg_costs.empty:
                rg_costs['pct_of_total'] = (rg_costs['total_cost'] / total_cost * 100).round(2)
                
                for _, row in rg_costs.head(10).iterrows():
                    top_resource_groups.append({
                        'resource_group': row['resource_group'],
                        'cost': round(float(row['total_cost']), 2),
                        'pct_of_total': float(row['pct_of_total']),
                        'record_count': int(row['record_count']),
                        'avg_cost': round(float(row['avg_cost']), 2)
                    })
                
                logger.info(
                    f"Top resource group: {top_resource_groups[0]['resource_group']} (${top_resource_groups[0]['cost']:,.2f})",
                    extra={'request_id': self.request_id}
                )
            
            # 5. Top cost drivers (resources)
            top_resources = get_top_n_resources(self.engine, month, n=10, request_id=self.request_id)
            
            cost_drivers = []
            if not top_resources.empty:
                for _, row in top_resources.iterrows():
                    cost_drivers.append({
                        'resource_id': row['resource_id'],
                        'service': row['service'],
                        'resource_group': row['resource_group'],
                        'region': row['region'],
                        'cost': round(float(row['total_cost']), 2),
                        'usage': float(row['total_usage']) if pd.notna(row['total_usage']) else None,
                        'owner': row['owner'] if pd.notna(row['owner']) else 'untagged',
                        'environment': row['env'] if pd.notna(row['env']) else 'untagged'
                    })
                
                logger.info(
                    f"Top cost driver: {cost_drivers[0]['resource_id']} (${cost_drivers[0]['cost']:,.2f})",
                    extra={'request_id': self.request_id}
                )
            
            # 6. Unit cost changes (if previous month exists)
            unit_cost_changes = []
            if not prev_monthly_costs.empty:
                unit_changes_df = get_unit_cost_changes(
                    self.engine, 
                    month, 
                    prev_month, 
                    self.request_id
                )
                
                if not unit_changes_df.empty:
                    for _, row in unit_changes_df.head(10).iterrows():
                        unit_cost_changes.append({
                            'resource_id': row['resource_id'],
                            'service': row['service'],
                            'resource_group': row['resource_group'],
                            'previous_unit_cost': round(float(row['previous_unit_cost']), 4),
                            'current_unit_cost': round(float(row['current_unit_cost']), 4),
                            'change_pct': round(float(row['unit_cost_change_pct']), 2),
                            'current_cost': round(float(row['current_cost']), 2)
                        })
                    
                    logger.info(
                        f"Found {len(unit_cost_changes)} resources with significant unit cost changes",
                        extra={'request_id': self.request_id}
                    )
            
            # Calculate duration
            kpi_duration = (datetime.now() - kpi_start).total_seconds()
            
            # Build response
            response = {
                'status': 'success',
                'month': month,
                'request_id': self.request_id,
                'calculation_time_seconds': round(kpi_duration, 3),
                'summary': {
                    'total_cost': round(total_cost, 2),
                    'record_count': record_count,
                    'unique_services': len(service_costs) if not service_costs.empty else 0,
                    'unique_resource_groups': len(rg_costs) if not rg_costs.empty else 0
                },
                'month_over_month': month_over_month,
                'top_services': top_services,
                'top_resource_groups': top_resource_groups,
                'top_cost_drivers': cost_drivers,
                'unit_cost_changes': unit_cost_changes
            }
            
            logger.info(
                f"✓ KPI calculation complete in {kpi_duration:.3f}s",
                extra={'request_id': self.request_id}
            )
            
            return response
            
        except Exception as e:
            logger.error(
                f"Error calculating KPIs: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def calculate_trend_kpis(self, months: int = 6) -> Dict:
        """
        Calculate trend KPIs over multiple months
        
        Args:
            months: Number of months to analyze
            
        Returns:
            dict: Trend analysis with growth metrics
        """
        logger.info(
            f"Calculating {months}-month trend KPIs",
            extra={'request_id': self.request_id}
        )
        
        try:
            # Get monthly trend
            trend_df = get_monthly_trend(self.engine, months, self.request_id)
            
            if trend_df.empty:
                logger.warning(
                    "No trend data available",
                    extra={'request_id': self.request_id}
                )
                return {
                    'status': 'no_data',
                    'message': 'Insufficient data for trend analysis'
                }
            
            # Build trend data
            trend_data = []
            for _, row in trend_df.iterrows():
                trend_point = {
                    'month': row['invoice_month'],
                    'cost': round(float(row['total_cost']), 2),
                    'record_count': int(row['record_count'])
                }
                
                if pd.notna(row.get('mom_change_pct')):
                    trend_point['mom_change_pct'] = round(float(row['mom_change_pct']), 2)
                
                trend_data.append(trend_point)
            
            # Calculate overall trend metrics
            first_month_cost = float(trend_df.iloc[0]['total_cost'])
            last_month_cost = float(trend_df.iloc[-1]['total_cost'])
            
            overall_change = last_month_cost - first_month_cost
            overall_change_pct = (overall_change / first_month_cost * 100) if first_month_cost > 0 else 0
            
            avg_monthly_cost = float(trend_df['total_cost'].mean())
            max_monthly_cost = float(trend_df['total_cost'].max())
            min_monthly_cost = float(trend_df['total_cost'].min())
            
            # Calculate average growth rate (excluding first month where it's NaN)
            if 'mom_change_pct' in trend_df.columns:
                avg_growth_rate = float(trend_df['mom_change_pct'].dropna().mean())
            else:
                avg_growth_rate = 0
            
            response = {
                'status': 'success',
                'request_id': self.request_id,
                'period': {
                    'months': len(trend_df),
                    'start_month': trend_df.iloc[0]['invoice_month'],
                    'end_month': trend_df.iloc[-1]['invoice_month']
                },
                'overall_metrics': {
                    'first_month_cost': round(first_month_cost, 2),
                    'last_month_cost': round(last_month_cost, 2),
                    'overall_change': round(overall_change, 2),
                    'overall_change_pct': round(overall_change_pct, 2),
                    'avg_monthly_cost': round(avg_monthly_cost, 2),
                    'avg_growth_rate_pct': round(avg_growth_rate, 2),
                    'max_monthly_cost': round(max_monthly_cost, 2),
                    'min_monthly_cost': round(min_monthly_cost, 2)
                },
                'trend_data': trend_data
            }
            
            logger.info(
                f"✓ Trend analysis complete: {overall_change_pct:+.2f}% over {len(trend_df)} months",
                extra={'request_id': self.request_id}
            )
            
            return response
            
        except Exception as e:
            logger.error(
                f"Error calculating trend KPIs: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def get_cost_breakdown_kpi(self, month: str, dimension: str = 'service') -> Dict:
        """
        Get detailed cost breakdown by specified dimension
        
        Args:
            month: Month in YYYY-MM format
            dimension: Dimension to break down by
            
        Returns:
            dict: Cost breakdown with percentages
        """
        logger.info(
            f"Getting cost breakdown by {dimension} for {month}",
            extra={'request_id': self.request_id}
        )
        
        try:
            breakdown_df = get_cost_breakdown(self.engine, month, dimension, self.request_id)
            
            if breakdown_df.empty:
                return {
                    'status': 'no_data',
                    'month': month,
                    'dimension': dimension,
                    'message': f'No data found for breakdown'
                }
            
            # Build breakdown list
            breakdown_list = []
            for _, row in breakdown_df.iterrows():
                breakdown_list.append({
                    dimension: row['dimension'],
                    'cost': round(float(row['total_cost']), 2),
                    'pct_of_total': float(row['pct_of_total']),
                    'record_count': int(row['record_count']),
                    'avg_cost': round(float(row['avg_cost']), 2),
                    'min_cost': round(float(row['min_cost']), 2),
                    'max_cost': round(float(row['max_cost']), 2)
                })
            
            total_cost = float(breakdown_df['total_cost'].sum())
            
            response = {
                'status': 'success',
                'month': month,
                'dimension': dimension,
                'request_id': self.request_id,
                'summary': {
                    'total_cost': round(total_cost, 2),
                    'unique_values': len(breakdown_df)
                },
                'breakdown': breakdown_list
            }
            
            logger.info(
                f"✓ Breakdown complete: {len(breakdown_df)} unique {dimension} values",
                extra={'request_id': self.request_id}
            )
            
            return response
            
        except Exception as e:
            logger.error(
                f"Error getting cost breakdown: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def get_tagging_compliance_kpi(self, month: str) -> Dict:
        """
        Calculate tagging compliance metrics
        
        Args:
            month: Month in YYYY-MM format
            
        Returns:
            dict: Tagging compliance metrics
        """
        logger.info(
            f"Calculating tagging compliance for {month}",
            extra={'request_id': self.request_id}
        )
        
        try:
            # Get resources without proper tags
            untagged_df = get_resources_without_tags(self.engine, month, self.request_id)
            
            # Get total resources and costs for the month
            total_resources_query = f"""
            SELECT 
                COUNT(DISTINCT resource_id) as total_resources,
                SUM(cost) as total_cost
            FROM billing
            WHERE invoice_month = '{month}'
            """
            
            total_df = pd.read_sql(total_resources_query, self.engine)
            total_resources = int(total_df['total_resources'].iloc[0])
            total_cost = float(total_df['total_cost'].iloc[0])
            
            # Calculate metrics
            untagged_resources = len(untagged_df)
            untagged_cost = float(untagged_df['total_cost'].sum()) if not untagged_df.empty else 0
            
            compliance_pct = ((total_resources - untagged_resources) / total_resources * 100) if total_resources > 0 else 0
            untagged_cost_pct = (untagged_cost / total_cost * 100) if total_cost > 0 else 0
            
            # Break down by missing tag type
            missing_tag_breakdown = {}
            if not untagged_df.empty:
                missing_tag_breakdown = {
                    'missing_owner': len(untagged_df[untagged_df['missing_tags'].isin(['owner', 'both'])]),
                    'missing_environment': len(untagged_df[untagged_df['missing_tags'].isin(['environment', 'both'])]),
                    'missing_both': len(untagged_df[untagged_df['missing_tags'] == 'both'])
                }
            
            # Top untagged resources
            top_untagged = []
            if not untagged_df.empty:
                for _, row in untagged_df.head(10).iterrows():
                    top_untagged.append({
                        'resource_id': row['resource_id'],
                        'service': row['service'],
                        'resource_group': row['resource_group'],
                        'cost': round(float(row['total_cost']), 2),
                        'missing_tags': row['missing_tags']
                    })
            
            response = {
                'status': 'success',
                'month': month,
                'request_id': self.request_id,
                'summary': {
                    'total_resources': total_resources,
                    'total_cost': round(total_cost, 2),
                    'untagged_resources': untagged_resources,
                    'untagged_cost': round(untagged_cost, 2),
                    'compliance_pct': round(compliance_pct, 2),
                    'untagged_cost_pct': round(untagged_cost_pct, 2)
                },
                'missing_tag_breakdown': missing_tag_breakdown,
                'top_untagged_resources': top_untagged
            }
            
            logger.info(
                f"✓ Tagging compliance: {compliance_pct:.1f}% ({untagged_resources} untagged resources)",
                extra={'request_id': self.request_id}
            )
            
            return response
            
        except Exception as e:
            logger.error(
                f"Error calculating tagging compliance: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise


def get_available_months(engine, request_id: str = "default") -> List[str]:
    """
    Get list of available months in the database
    
    Args:
        engine: SQLAlchemy engine
        request_id: Request ID for logging
        
    Returns:
        list: List of months in YYYY-MM format
    """
    logger.info(
        "Fetching available months",
        extra={'request_id': request_id}
    )
    
    try:
        query = "SELECT DISTINCT invoice_month FROM billing ORDER BY invoice_month DESC"
        df = pd.read_sql(query, engine)
        
        months = df['invoice_month'].tolist()
        
        logger.info(
            f"Found {len(months)} available months: {months}",
            extra={'request_id': request_id}
        )
        
        return months
        
    except Exception as e:
        logger.error(
            f"Error fetching available months: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise