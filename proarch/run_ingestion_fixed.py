#!/usr/bin/env python3
"""
Simple script to run data ingestion with explicit commits
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ingestion.loader import BillingDataLoader

if __name__ == '__main__':
    # Load environment variables
    load_dotenv()

    # Set logging to INFO level
    logging.basicConfig(level=logging.INFO)

    # Get configuration
    db_url = os.getenv('DATABASE_URL', 'sqlite:///./data/cost_analytics.db')
    csv_path = os.getenv('BILLING_CSV_PATH', 'data/raw/billing_raw.csv')

    # Create loader and run ingestion
    loader = BillingDataLoader(db_url, request_id="manual-ingest")

    try:
        result = loader.ingest(csv_path)
        print("SUCCESS: Ingestion completed!")
        print(f"Billing records: {result['metrics']['billing_records_loaded']}")
        print(f"Resource records: {result['metrics']['resource_records_loaded']}")
    except Exception as e:
        print(f"FAILED: {str(e)}")
        sys.exit(1)
