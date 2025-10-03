"""
Unit tests for ETL (Extract, Transform, Load) functionality
Tests data ingestion, transformation, and database operations
"""

import pytest
import pandas as pd
import tempfile
import os
import sys
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from ingestion.loader import BillingDataLoader
    from ingestion.transformers import transform_to_schema
    from ingestion.database_operations import load_to_database
    from ingestion.schema import init_db, get_engine
    from transformations.kpis import get_available_months
except ImportError:
    # Fallback for pytest execution
    import sys
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from ingestion.loader import BillingDataLoader
    from ingestion.transformers import transform_to_schema
    from ingestion.database_operations import load_to_database
    from ingestion.schema import init_db, get_engine
    from transformations.kpis import get_available_months


class TestETL:
    """Test ETL pipeline components"""

    def setup_method(self):
        """Setup for each test method"""
        # Create temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_url = f"sqlite:///{self.temp_db.name}"

        # Create sample CSV data
        self.sample_csv_data = {
            'AvailabilityZone': ['us-west-2b', 'us-east-1a'],
            'BilledCost': [100.50, 75.25],
            'BillingAccountId': ['1234567890123', '1234567890123'],
            'BillingAccountName': ['TestAccount', 'TestAccount'],
            'BillingCurrency': ['USD', 'USD'],
            'BillingPeriodEnd': ['2024-10-01 00:00:00', '2024-10-01 00:00:00'],
            'BillingPeriodStart': ['2024-09-01 00:00:00', '2024-09-01 00:00:00'],
            'ChargeCategory': ['Usage', 'Usage'],
            'ConsumedQuantity': [10.0, 5.0],
            'ConsumedUnit': ['Hours', 'Hours'],
            'EffectiveCost': [100.50, 75.25],
            'ProviderName': ['AWS', 'AWS'],
            'RegionId': ['us-west-2', 'us-east-1'],
            'RegionName': ['US West (Oregon)', 'US East (N. Virginia)'],
            'ResourceId': ['i-1234567890abcdef0', 'i-0987654321fedcba0'],
            'ServiceName': ['Amazon Elastic Compute Cloud', 'Amazon Elastic Compute Cloud'],
            'Tags': ['{"environment": "prod", "team": "engineering"}', '{"environment": "dev", "team": "qa"}']
        }
        self.sample_df = pd.DataFrame(self.sample_csv_data)

    def teardown_method(self):
        """Cleanup after each test method"""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass

    def test_billing_data_loader_initialization(self):
        """Test BillingDataLoader initialization"""
        loader = BillingDataLoader(self.db_url, request_id="test-init")

        assert loader.db_url == self.db_url
        assert loader.request_id == "test-init"
        assert hasattr(loader, 'engine')
        assert hasattr(loader, 'Session')

    def test_csv_loading(self):
        """Test CSV file loading functionality"""
        # Create temporary CSV file
        temp_csv = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        self.sample_df.to_csv(temp_csv.name, index=False)
        temp_csv.close()

        try:
            loader = BillingDataLoader(self.db_url, request_id="test-csv")

            # Test loading
            df = loader.load_csv(temp_csv.name)

            assert len(df) == 2
            assert 'BilledCost' in df.columns
            assert df['BilledCost'].sum() == 175.75

        finally:
            os.unlink(temp_csv.name)

    def test_data_transformation(self):
        """Test data transformation from raw to schema format"""
        # Transform the sample data
        billing_df, resources_df, parse_errors, rows_skipped = transform_to_schema(
            self.sample_df, "test-transform"
        )

        # Assertions
        assert len(billing_df) == 2  # Should have 2 billing records
        assert len(resources_df) == 2  # Should have 2 unique resources

        # Check billing schema columns
        expected_billing_cols = ['id', 'invoice_month', 'account_id', 'subscription',
                               'service', 'resource_group', 'resource_id', 'region',
                               'usage_qty', 'unit_cost', 'cost']
        for col in expected_billing_cols:
            assert col in billing_df.columns

        # Check resources schema columns
        expected_resource_cols = ['id', 'resource_id', 'owner', 'env', 'tags_json']
        for col in expected_resource_cols:
            assert col in resources_df.columns

        # Check data integrity
        assert billing_df['cost'].sum() == 175.75
        assert parse_errors == 0
        assert rows_skipped == 0

    def test_database_operations(self):
        """Test database loading operations"""
        # Initialize database
        engine = get_engine(self.db_url)
        init_db(self.db_url)

        # Transform data first
        billing_df, resources_df, _, _ = transform_to_schema(self.sample_df, "test-db")

        # Load to database
        billing_loaded, resources_loaded = load_to_database(billing_df, resources_df, engine, "test-db")

        assert billing_loaded == 2
        assert resources_loaded == 2

        # Verify data was loaded
        with engine.connect() as conn:
            result = conn.execute("SELECT COUNT(*) FROM billing")
            billing_count = result.scalar()
            assert billing_count == 2

            result = conn.execute("SELECT COUNT(*) FROM resources")
            resources_count = result.scalar()
            assert resources_count == 2

    @patch('ingestion.database_operations.inspect')
    def test_database_error_handling(self, mock_inspect):
        """Test database error handling"""
        # Mock inspect to raise an exception
        mock_inspect.side_effect = Exception("Database connection error")

        engine = get_engine(self.db_url)

        with pytest.raises(Exception):
            load_to_database(pd.DataFrame(), pd.DataFrame(), engine, "test-error")

    def test_etl_pipeline_integration(self):
        """Test full ETL pipeline integration"""
        # Create temporary CSV file
        temp_csv = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        self.sample_df.to_csv(temp_csv.name, index=False)
        temp_csv.close()

        try:
            # Run full pipeline
            loader = BillingDataLoader(self.db_url, request_id="test-pipeline")

            result = loader.ingest(temp_csv.name, skip_quality_check=True)

            # Verify results
            assert result['status'] == 'success'
            assert result['metrics']['billing_records_loaded'] == 2
            assert result['metrics']['resource_records_loaded'] == 2
            assert result['metrics']['rows_read'] == 2
            assert result['metrics']['rows_transformed'] == 2

        finally:
            os.unlink(temp_csv.name)
