"""
Anomalies Module
Detects cost anomalies, spikes, and unusual patterns in cloud spending
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy import stats

from .aggregations import (
    get_monthly_trend,
    get_costs_by_service,
    get_top_n_resources,
    get_unit_cost_changes
)

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects cost anomalies and unusual spending patterns"""
    
    def __init__(self, engine, request_id: str = "default"):
        self.engine = engine
        self.request_id = request_id
        
        logger.info(
            "AnomalyDetector initialized",
            extra={'request_id': request_id}
        )
    
    def detect_cost_spikes(self, months: int = 6, threshold_std: float = 2.0) -> Dict:
        """
        Detect months with unusual cost spikes using statistical methods
        
        Args:
            months: Number of months to analyze
            threshold_std: Standard deviation threshold for anomaly detection
            
        Returns:
            dict: Detected anomalies with details
        """
        logger.info(
            f"Detecting cost spikes over {months} months (threshold: {threshold_std} std)",
            extra={'request_id': self.request_id}
        )
        
        try:
            # Get trend data
            trend_df = get_monthly_trend(self.engine, months, self.request_id)
            
            if len(trend_df) < 3:
                logger.warning(
                    f"Insufficient data for anomaly detection (need at least 3 months, got {len(trend_df)})",
                    extra={'request_id': self.request_id}
                )
                return {
                    'status': 'insufficient_data',
                    'message': 'Need at least 3 months of data for anomaly detection'
                }
            
            # Calculate statistics
            costs = trend_df['total_cost'].values
            mean_cost = np.mean(costs)
            std_cost = np.std(costs)
            
            # Detect anomalies using z-score
            z_scores = np.abs((costs - mean_cost) / std_cost) if std_cost > 0 else np.zeros(len(costs))
            
            anomalies = []
            for idx, (_, row) in enumerate(trend_df.iterrows()):
                z_score = z_scores[idx]
                
                if z_score >= threshold_std:
                    cost = float(row['total_cost'])
                    deviation = cost - mean_cost
                    deviation_pct = (deviation / mean_cost * 100) if mean_cost > 0 else 0
                    
                    anomaly = {
                        'month': row['invoice_month'],
                        'cost': round(cost, 2),
                        'expected_cost': round(mean_cost, 2),
                        'deviation': round(deviation, 2),
                        'deviation_pct': round(deviation_pct, 2),
                        'z_score': round(float(z_score), 2),
                        'severity': self._classify_severity(z_score, threshold_std),
                        'type': 'spike' if deviation > 0 else 'drop'
                    }
                    
                    anomalies.append(anomaly)
                    
                    logger.warning(
                        f"⚠️  Anomaly detected in {row['invoice_month']}: ${cost:,.2f} "
                        f"(z-score: {z_score:.2f}, deviation: {deviation_pct:+.1f}%)",
                        extra={'request_id': self.request_id}
                    )
            
            response = {
                'status': 'success',
                'request_id': self.request_id,
                'analysis_period': {
                    'months_analyzed': len(trend_df),
                    'start_month': trend_df.iloc[0]['invoice_month'],
                    'end_month': trend_df.iloc[-1]['invoice_month']
                },
                'baseline': {
                    'mean_cost': round(mean_cost, 2),
                    'std_cost': round(std_cost, 2),
                    'threshold_std': threshold_std
                },
                'anomalies_detected': len(anomalies),
                'anomalies': sorted(anomalies, key=lambda x: abs(x['z_score']), reverse=True)
            }
            
            logger.info(
                f"✓ Spike detection complete: {len(anomalies)} anomalies found",
                extra={'request_id': self.request_id}
            )
            
            return response
            
        except Exception as e:
            logger.error(
                f"Error detecting cost spikes: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def detect_service_anomalies(self, month: str, previous_month: str) -> Dict:
        """
        Detect unusual service-level cost changes between two months
        
        Args:
            month: Current month in YYYY-MM format
            previous_month: Previous month in YYYY-MM format
            
        Returns:
            dict: Service-level anomalies
        """
        logger.info(
            f"Detecting service anomalies between {previous_month} and {month}",
            extra={'request_id': self.request_id}
        )
        
        try:
            # Get service costs for both months
            current_services = get_costs_by_service(self.engine, month, self.request_id)
            previous_services = get_costs_by_service(self.engine, previous_month, self.request_id)
            
            if current_services.empty or previous_services.empty:
                logger.warning(
                    "Insufficient data for service anomaly detection",
                    extra={'request_id': self.request_id}
                )
                return {
                    'status': 'insufficient_data',
                    'message': 'Need data for both months'
                }
            
            # Merge to compare
            merged = current_services.merge(
                previous_services,
                on='service',
                how='outer',
                suffixes=('_current', '_previous')
            )
            
            # Fill NaN with 0 for new or removed services
            merged['total_cost_current'] = merged['total_cost_current'].fillna(0)
            merged['total_cost_previous'] = merged['total_cost_previous'].fillna(0)
            
            # Calculate changes
            merged['absolute_change'] = merged['total_cost_current'] - merged['total_cost_previous']
            merged['percent_change'] = np.where(
                merged['total_cost_previous'] > 0,
                (merged['absolute_change'] / merged['total_cost_previous'] * 100),
                np.inf  # New services
            )
            
            # Detect anomalies (significant changes)
            anomaly_threshold = 30  # 30% change
            
            anomalies = []
            
            for _, row in merged.iterrows():
                service = row['service']
                current_cost = float(row['total_cost_current'])
                previous_cost = float(row['total_cost_previous'])
                change = float(row['absolute_change'])
                change_pct = float(row['percent_change']) if not np.isinf(row['percent_change']) else None
                
                # New service
                if previous_cost == 0 and current_cost > 0:
                    anomalies.append({
                        'service': service,
                        'type': 'new_service',
                        'current_cost': round(current_cost, 2),
                        'previous_cost': 0,
                        'change': round(change, 2),
                        'severity': 'info',
                        'message': f"New service detected: {service}"
                    })
                    continue
                
                # Removed service
                if current_cost == 0 and previous_cost > 0:
                    anomalies.append({
                        'service': service,
                        'type': 'removed_service',
                        'current_cost': 0,
                        'previous_cost': round(previous_cost, 2),
                        'change': round(change, 2),
                        'severity': 'warning',
                        'message': f"Service no longer active: {service}"
                    })
                    continue
                
                # Significant increase
                if change_pct and change_pct >= anomaly_threshold:
                    severity = 'high' if change_pct >= 50 else 'medium'
                    anomalies.append({
                        'service': service,
                        'type': 'spike',
                        'current_cost': round(current_cost, 2),
                        'previous_cost': round(previous_cost, 2),
                        'change': round(change, 2),
                        'change_pct': round(change_pct, 2),
                        'severity': severity,
                        'message': f"{service} cost increased by {change_pct:.1f}%"
                    })
                
                # Significant decrease
                elif change_pct and change_pct <= -anomaly_threshold:
                    severity = 'high' if change_pct <= -50 else 'medium'
                    anomalies.append({
                        'service': service,
                        'type': 'drop',
                        'current_cost': round(current_cost, 2),
                        'previous_cost': round(previous_cost, 2),
                        'change': round(change, 2),
                        'change_pct': round(change_pct, 2),
                        'severity': severity,
                        'message': f"{service} cost decreased by {abs(change_pct):.1f}%"
                    })
            
            # Sort by absolute change
            anomalies = sorted(anomalies, key=lambda x: abs(x.get('change', 0)), reverse=True)
            
            response = {
                'status': 'success',
                'request_id': self.request_id,
                'current_month': month,
                'previous_month': previous_month,
                'anomalies_detected': len(anomalies),
                'anomalies': anomalies
            }
            
            logger.info(
                f"✓ Service anomaly detection complete: {len(anomalies)} anomalies found",
                extra={'request_id': self.request_id}
            )
            
            return response
            
        except Exception as e:
            logger.error(
                f"Error detecting service anomalies: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def detect_resource_anomalies(self, month: str, top_n: int = 20) -> Dict:
        """
        Detect unusual resource-level spending patterns
        
        Args:
            month: Month in YYYY-MM format
            top_n: Number of top resources to analyze
            
        Returns:
            dict: Resource-level anomalies
        """
        logger.info(
            f"Detecting resource anomalies for {month} (analyzing top {top_n} resources)",
            extra={'request_id': self.request_id}
        )
        
        try:
            # Get top resources
            top_resources = get_top_n_resources(self.engine, month, n=top_n, request_id=self.request_id)
            
            if top_resources.empty:
                logger.warning(
                    f"No resource data found for {month}",
                    extra={'request_id': self.request_id}
                )
                return {
                    'status': 'no_data',
                    'message': f'No resource data found for {month}'
                }
            
            # Get historical data for these resources
            resource_ids = top_resources['resource_id'].tolist()
            resource_ids_str = "', '".join(resource_ids)
            
            history_query = f"""
            SELECT 
                resource_id,
                invoice_month,
                SUM(cost) as total_cost,
                SUM(usage_qty) as total_usage
            FROM billing
            WHERE resource_id IN ('{resource_ids_str}')
            GROUP BY resource_id, invoice_month
            ORDER BY resource_id, invoice_month
            """
            
            history_df = pd.read_sql(history_query, self.engine)
            
            anomalies = []
            
            # Analyze each resource
            for resource_id in resource_ids:
                resource_history = history_df[history_df['resource_id'] == resource_id].copy()
                
                if len(resource_history) < 2:
                    continue  # Need at least 2 months for comparison
                
                # Get current month data
                current_data = resource_history[resource_history['invoice_month'] == month]
                
                if current_data.empty:
                    continue
                
                current_cost = float(current_data['total_cost'].iloc[0])
                
                # Calculate historical baseline (excluding current month)
                historical_data = resource_history[resource_history['invoice_month'] != month]
                
                if len(historical_data) > 0:
                    avg_historical_cost = float(historical_data['total_cost'].mean())
                    std_historical_cost = float(historical_data['total_cost'].std())
                    
                    # Detect anomaly
                    if std_historical_cost > 0:
                        z_score = (current_cost - avg_historical_cost) / std_historical_cost
                        
                        if abs(z_score) >= 2.0:  # 2 standard deviations
                            resource_info = top_resources[top_resources['resource_id'] == resource_id].iloc[0]
                            
                            deviation = current_cost - avg_historical_cost
                            deviation_pct = (deviation / avg_historical_cost * 100) if avg_historical_cost > 0 else 0
                            
                            anomaly = {
                                'resource_id': resource_id,
                                'service': resource_info['service'],
                                'resource_group': resource_info['resource_group'],
                                'current_cost': round(current_cost, 2),
                                'avg_historical_cost': round(avg_historical_cost, 2),
                                'deviation': round(deviation, 2),
                                'deviation_pct': round(deviation_pct, 2),
                                'z_score': round(float(z_score), 2),
                                'months_analyzed': len(historical_data) + 1,
                                'severity': self._classify_severity(abs(z_score), 2.0),
                                'type': 'spike' if deviation > 0 else 'drop'
                            }
                            
                            anomalies.append(anomaly)
            
            # Sort by z-score
            anomalies = sorted(anomalies, key=lambda x: abs(x['z_score']), reverse=True)
            
            response = {
                'status': 'success',
                'request_id': self.request_id,
                'month': month,
                'resources_analyzed': top_n,
                'anomalies_detected': len(anomalies),
                'anomalies': anomalies
            }
            
            logger.info(
                f"✓ Resource anomaly detection complete: {len(anomalies)} anomalies found among {top_n} resources",
                extra={'request_id': self.request_id}
            )
            
            return response
            
        except Exception as e:
            logger.error(
                f"Error detecting resource anomalies: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def detect_sudden_usage_changes(self, month: str, previous_month: str) -> Dict:
        """
        Detect resources with sudden usage pattern changes
        
        Args:
            month: Current month in YYYY-MM format
            previous_month: Previous month in YYYY-MM format
            
        Returns:
            dict: Usage pattern changes
        """
        logger.info(
            f"Detecting usage changes between {previous_month} and {month}",
            extra={'request_id': self.request_id}
        )
        
        try:
            query = f"""
            WITH current AS (
                SELECT 
                    resource_id,
                    service,
                    resource_group,
                    SUM(usage_qty) as current_usage,
                    SUM(cost) as current_cost,
                    AVG(unit_cost) as current_unit_cost
                FROM billing
                WHERE invoice_month = '{month}'
                GROUP BY resource_id, service, resource_group
            ),
            previous AS (
                SELECT 
                    resource_id,
                    SUM(usage_qty) as previous_usage,
                    SUM(cost) as previous_cost,
                    AVG(unit_cost) as previous_unit_cost
                FROM billing
                WHERE invoice_month = '{previous_month}'
                GROUP BY resource_id
            )
            SELECT 
                c.resource_id,
                c.service,
                c.resource_group,
                c.current_usage,
                c.current_cost,
                c.current_unit_cost,
                p.previous_usage,
                p.previous_cost,
                p.previous_unit_cost
            FROM current c
            INNER JOIN previous p ON c.resource_id = p.resource_id
            WHERE c.current_usage IS NOT NULL 
            AND p.previous_usage IS NOT NULL
            AND p.previous_usage > 0
            """
            
            df = pd.read_sql(query, self.engine)
            
            if df.empty:
                logger.warning(
                    "No usage data for comparison",
                    extra={'request_id': self.request_id}
                )
                return {
                    'status': 'no_data',
                    'message': 'Insufficient usage data for comparison'
                }
            
            # Calculate changes
            df['usage_change'] = df['current_usage'] - df['previous_usage']
            df['usage_change_pct'] = (df['usage_change'] / df['previous_usage'] * 100)
            df['cost_change'] = df['current_cost'] - df['previous_cost']
            df['cost_change_pct'] = (df['cost_change'] / df['previous_cost'] * 100)
            
            # Detect anomalies (>50% usage change)
            threshold = 50
            anomalies_df = df[abs(df['usage_change_pct']) > threshold].copy()
            anomalies_df = anomalies_df.sort_values('usage_change_pct', ascending=False)
            
            anomalies = []
            for _, row in anomalies_df.head(20).iterrows():
                anomaly = {
                    'resource_id': row['resource_id'],
                    'service': row['service'],
                    'resource_group': row['resource_group'],
                    'current_usage': round(float(row['current_usage']), 2),
                    'previous_usage': round(float(row['previous_usage']), 2),
                    'usage_change_pct': round(float(row['usage_change_pct']), 2),
                    'current_cost': round(float(row['current_cost']), 2),
                    'previous_cost': round(float(row['previous_cost']), 2),
                    'cost_change_pct': round(float(row['cost_change_pct']), 2),
                    'type': 'usage_spike' if row['usage_change_pct'] > 0 else 'usage_drop',
                    'severity': 'high' if abs(row['usage_change_pct']) > 100 else 'medium'
                }
                anomalies.append(anomaly)
            
            response = {
                'status': 'success',
                'request_id': self.request_id,
                'current_month': month,
                'previous_month': previous_month,
                'resources_analyzed': len(df),
                'anomalies_detected': len(anomalies),
                'threshold_pct': threshold,
                'anomalies': anomalies
            }
            
            logger.info(
                f"✓ Usage change detection complete: {len(anomalies)} significant changes found",
                extra={'request_id': self.request_id}
            )
            
            return response
            
        except Exception as e:
            logger.error(
                f"Error detecting usage changes: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def detect_all_anomalies(self, month: str, previous_month: str) -> Dict:
        """
        Run all anomaly detection methods and return comprehensive report
        
        Args:
            month: Current month in YYYY-MM format
            previous_month: Previous month in YYYY-MM format
            
        Returns:
            dict: Complete anomaly report
        """
        logger.info(
            f"Running comprehensive anomaly detection for {month}",
            extra={'request_id': self.request_id}
        )
        
        detection_start = datetime.now()
        
        try:
            # 1. Cost spikes
            cost_spikes = self.detect_cost_spikes(months=6, threshold_std=2.0)
            
            # 2. Service anomalies
            service_anomalies = self.detect_service_anomalies(month, previous_month)
            
            # 3. Resource anomalies
            resource_anomalies = self.detect_resource_anomalies(month, top_n=20)
            
            # 4. Usage changes
            usage_changes = self.detect_sudden_usage_changes(month, previous_month)
            
            # 5. Unit cost changes
            unit_cost_changes = get_unit_cost_changes(
                self.engine, 
                month, 
                previous_month, 
                self.request_id
            )
            
            unit_cost_anomalies = []
            if not unit_cost_changes.empty:
                for _, row in unit_cost_changes.head(10).iterrows():
                    if abs(row['unit_cost_change_pct']) > 20:  # >20% change
                        unit_cost_anomalies.append({
                            'resource_id': row['resource_id'],
                            'service': row['service'],
                            'resource_group': row['resource_group'],
                            'previous_unit_cost': round(float(row['previous_unit_cost']), 4),
                            'current_unit_cost': round(float(row['current_unit_cost']), 4),
                            'change_pct': round(float(row['unit_cost_change_pct']), 2),
                            'current_cost': round(float(row['current_cost']), 2),
                            'type': 'unit_cost_increase' if row['unit_cost_change_pct'] > 0 else 'unit_cost_decrease',
                            'severity': 'high' if abs(row['unit_cost_change_pct']) > 50 else 'medium'
                        })
            
            # Calculate summary statistics
            total_anomalies = (
                cost_spikes.get('anomalies_detected', 0) +
                service_anomalies.get('anomalies_detected', 0) +
                resource_anomalies.get('anomalies_detected', 0) +
                usage_changes.get('anomalies_detected', 0) +
                len(unit_cost_anomalies)
            )
            
            detection_duration = (datetime.now() - detection_start).total_seconds()
            
            response = {
                'status': 'success',
                'request_id': self.request_id,
                'month': month,
                'previous_month': previous_month,
                'detection_time_seconds': round(detection_duration, 3),
                'summary': {
                    'total_anomalies': total_anomalies,
                    'cost_spikes': cost_spikes.get('anomalies_detected', 0),
                    'service_anomalies': service_anomalies.get('anomalies_detected', 0),
                    'resource_anomalies': resource_anomalies.get('anomalies_detected', 0),
                    'usage_changes': usage_changes.get('anomalies_detected', 0),
                    'unit_cost_anomalies': len(unit_cost_anomalies)
                },
                'cost_spikes': cost_spikes,
                'service_anomalies': service_anomalies,
                'resource_anomalies': resource_anomalies,
                'usage_changes': usage_changes,
                'unit_cost_anomalies': unit_cost_anomalies
            }
            
            logger.info(
                f"✓ Complete anomaly detection finished in {detection_duration:.3f}s: "
                f"{total_anomalies} total anomalies detected",
                extra={'request_id': self.request_id}
            )
            
            return response
            
        except Exception as e:
            logger.error(
                f"Error in comprehensive anomaly detection: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    @staticmethod
    def _classify_severity(z_score: float, threshold: float) -> str:
        """
        Classify anomaly severity based on z-score
        
        Args:
            z_score: Absolute z-score value
            threshold: Base threshold
            
        Returns:
            str: Severity level (low, medium, high, critical)
        """
        if z_score >= threshold * 2:
            return 'critical'
        elif z_score >= threshold * 1.5:
            return 'high'
        elif z_score >= threshold:
            return 'medium'
        else:
            return 'low'
        
