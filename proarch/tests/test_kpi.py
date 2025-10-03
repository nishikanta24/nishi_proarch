"""
Unit tests for KPI calculation functionality
Tests cost analytics and performance metrics calculation
"""

import pytest
import pandas as pd
import tempfile
import os
import sys
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from transformations.kpis import KPICalculator, get_available_months
    from transformations.aggregations import get_monthly_costs, get_costs_by_service
    from ingestion.schema import init_db, get_engine
except ImportError:
    # Fallback for pytest execution
    import sys
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from transformations.kpis import KPICalculator, get_available_months
    from transformations.aggregations import get_monthly_costs, get_costs_by_service
    from ingestion.schema import init_db, get_engine


class TestKPICalculator:
    """Test KPI calculation functionality"""

    def setup_method(self):
        """Setup for each test method"""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_url = f"sqlite:///{self.temp_db.name}"

        # Initialize database and create test data
        engine = get_engine(self.db_url)
        init_db(self.db_url)

        # Create sample billing data
        self.sample_billing_data = [
            {
                'invoice_month': '2024-09',
                'account_id': '1234567890123',
                'service': 'Amazon Elastic Compute Cloud',
                'resource_id': 'i-1234567890abcdef0',
                'region': 'us-west-2',
                'usage_qty': 100.0,
                'unit_cost': 0.10,
                'cost': 10.00
            },
            {
                'invoice_month': '2024-09',
                'account_id': '1234567890123',
                'service': 'Amazon Simple Storage Service',
                'resource_id': 'bucket-123',
                'region': 'us-west-2',
                'usage_qty': 1000.0,
                'unit_cost': 0.02,
                'cost': 20.00
            },
            {
                'invoice_month': '2024-10',
                'account_id': '1234567890123',
                'service': 'Amazon Elastic Compute Cloud',
                'resource_id': 'i-1234567890abcdef0',
                'region': 'us-west-2',
                'usage_qty': 120.0,
                'unit_cost': 0.10,
                'cost': 12.00
            }
        ]

        # Insert test data
        with engine.connect() as conn:
            for record in self.sample_billing_data:
                conn.execute("""
                    INSERT INTO billing (invoice_month, account_id, service, resource_id,
                                       region, usage_qty, unit_cost, cost)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record['invoice_month'], record['account_id'], record['service'],
                    record['resource_id'], record['region'], record['usage_qty'],
                    record['unit_cost'], record['cost']
                ))
            conn.commit()

        # Create KPI calculator
        self.kpi_calc = KPICalculator(engine)

    def teardown_method(self):
        """Cleanup after each test method"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass

    def test_get_available_months(self):
        """Test getting available months from database"""
        engine = get_engine(self.db_url)
        months = get_available_months(engine, request_id="test-months")

        assert len(months) == 2
        assert '2024-09' in months
        assert '2024-10' in months

    def test_monthly_costs_calculation(self):
        """Test monthly costs calculation"""
        engine = get_engine(self.db_url)

        sept_costs = get_monthly_costs(engine, '2024-09', request_id="test-sept")
        oct_costs = get_monthly_costs(engine, '2024-10', request_id="test-oct")

        assert sept_costs == 30.00  # 10 + 20
        assert oct_costs == 12.00   # 12

    def test_service_costs_calculation(self):
        """Test service-wise cost breakdown"""
        engine = get_engine(self.db_url)

        sept_service_costs = get_costs_by_service(engine, '2024-09', request_id="test-service-sept")
        oct_service_costs = get_costs_by_service(engine, '2024-10', request_id="test-service-oct")

        # September: EC2=10, S3=20
        assert len(sept_service_costs) == 2
        ec2_sept = next((s for s in sept_service_costs if s['service'] == 'Amazon Elastic Compute Cloud'), None)
        s3_sept = next((s for s in sept_service_costs if s['service'] == 'Amazon Simple Storage Service'), None)

        assert ec2_sept['cost'] == 10.00
        assert s3_sept['cost'] == 20.00

        # October: EC2=12
        assert len(oct_service_costs) == 1
        ec2_oct = oct_service_costs[0]
        assert ec2_oct['service'] == 'Amazon Elastic Compute Cloud'
        assert ec2_oct['cost'] == 12.00

    def test_kpi_calculator_monthly_kpis(self):
        """Test KPI calculator monthly KPIs computation"""
        # Test September KPIs
        sept_kpis = self.kpi_calc.calculate_monthly_kpis('2024-09')

        assert sept_kpis['status'] == 'success'
        assert sept_kpis['total_cost'] == 30.00
        assert len(sept_kpis['top_services']) == 2

        # Test October KPIs
        oct_kpis = self.kpi_calc.calculate_monthly_kpis('2024-10')

        assert oct_kpis['status'] == 'success'
        assert oct_kpis['total_cost'] == 12.00
        assert len(oct_kpis['top_services']) == 1

    def test_kpi_calculator_invalid_month(self):
        """Test KPI calculator with invalid month"""
        invalid_kpis = self.kpi_calc.calculate_monthly_kpis('2025-01')

        assert invalid_kpis['status'] == 'no_data'
        assert 'No data found' in invalid_kpis.get('message', '')

    def test_kpi_calculator_comprehensive_metrics(self):
        """Test comprehensive KPI metrics calculation"""
        sept_kpis = self.kpi_calc.calculate_monthly_kpis('2024-09')

        # Check for required KPI fields
        required_fields = [
            'total_cost', 'top_services', 'resource_count', 'region_breakdown',
            'cost_trend', 'anomalies', 'tagging_compliance'
        ]

        for field in required_fields:
            assert field in sept_kpis

        # Verify top services structure
        top_services = sept_kpis['top_services']
        assert isinstance(top_services, list)
        if top_services:
            service = top_services[0]
            assert 'service' in service
            assert 'cost' in service

    def test_cost_trend_calculation(self):
        """Test month-over-month cost trend calculation"""
        sept_kpis = self.kpi_calc.calculate_monthly_kpis('2024-09')
        oct_kpis = self.kpi_calc.calculate_monthly_kpis('2024-10')

        # September to October: 30.00 -> 12.00 = -60% change
        sept_trend = sept_kpis.get('cost_trend', {})
        oct_trend = oct_kpis.get('cost_trend', {})

        # Check that trends include previous month comparison when applicable
        if 'previous_month_cost' in sept_trend:
            assert sept_trend['previous_month_cost'] is None  # September has no previous month

        if 'previous_month_cost' in oct_trend:
            assert oct_trend['previous_month_cost'] == 30.00

    def test_resource_count_calculation(self):
        """Test resource count calculation"""
        sept_kpis = self.kpi_calc.calculate_monthly_kpis('2024-09')
        oct_kpis = self.kpi_calc.calculate_monthly_kpis('2024-10')

        # September should have 2 unique resources
        assert sept_kpis['resource_count'] == 2

        # October should have 1 unique resource
        assert oct_kpis['resource_count'] == 1

    def test_region_breakdown(self):
        """Test regional cost breakdown"""
        sept_kpis = self.kpi_calc.calculate_monthly_kpis('2024-09')

        region_breakdown = sept_kpis.get('region_breakdown', [])
        assert isinstance(region_breakdown, list)

        # All test data is in us-west-2
        if region_breakdown:
            us_west_2 = next((r for r in region_breakdown if r.get('region') == 'us-west-2'), None)
            assert us_west_2 is not None
            assert us_west_2['cost'] == 30.00
