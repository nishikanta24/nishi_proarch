"""
Data Ingestion Module
Orchestrates the complete data ingestion pipeline.
Includes comprehensive logging, metrics tracking, and error handling.
"""

import pandas as pd
import logging
import os
from datetime import datetime
from typing import Dict
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect

from .schema import Billing, Resource, init_db, get_engine
from .metrics import IngestionMetrics
from .transformers import transform_to_schema
from .quality_checks import run_quality_checks, DataQualityError
from .database_operations import load_to_database, sanitize_db_url

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BillingDataLoader:
    """Handles ingestion and transformation of billing data"""
    
    def __init__(self, db_url: str, request_id: str = "init"):
        self.db_url = db_url
        self.request_id = request_id
        self.metrics = IngestionMetrics()
        
        # Log configuration
        logger.info(
            f"Initializing BillingDataLoader with database: {sanitize_db_url(db_url)}",
            extra={'request_id': request_id}
        )
        
        try:
            self.engine = get_engine(db_url)
            self.Session = sessionmaker(bind=self.engine)
            
            # Check if tables exist
            inspector = inspect(self.engine)
            existing_tables = inspector.get_table_names()
            logger.info(
                f"Existing tables in database: {existing_tables if existing_tables else 'None (new database)'}",
                extra={'request_id': request_id}
            )
            
            # Initialize database tables
            init_db(db_url)
            logger.info(
                f"Database tables initialized successfully",
                extra={'request_id': request_id}
            )
            
        except Exception as e:
            logger.error(
                f"Failed to initialize database: {str(e)}",
                extra={'request_id': request_id},
                exc_info=True
            )
            raise
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV file into pandas DataFrame with comprehensive logging
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            Exception: For other loading errors
        """
        logger.info(
            f"Attempting to load CSV from path: {os.path.abspath(filepath)}",
            extra={'request_id': self.request_id}
        )
        
        # Check file existence
        if not os.path.exists(filepath):
            logger.error(
                f"CSV file not found at path: {os.path.abspath(filepath)}",
                extra={'request_id': self.request_id}
            )
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        # Check file is readable
        if not os.access(filepath, os.R_OK):
            logger.error(
                f"CSV file exists but is not readable: {os.path.abspath(filepath)}",
                extra={'request_id': self.request_id}
            )
            raise PermissionError(f"CSV file not readable: {filepath}")
        
        # Log file size
        file_size = os.path.getsize(filepath)
        logger.info(
            f"CSV file size: {file_size / (1024*1024):.2f} MB",
            extra={'request_id': self.request_id}
        )
        
        try:
            load_start = datetime.now()
            df = pd.read_csv(filepath)
            load_duration = (datetime.now() - load_start).total_seconds()
            
            self.metrics.rows_read = len(df)
            
            logger.info(
                f"Successfully loaded CSV: {len(df)} rows, {len(df.columns)} columns in {load_duration:.2f}s",
                extra={'request_id': self.request_id}
            )
            logger.info(
                f"Columns found: {list(df.columns)}",
                extra={'request_id': self.request_id}
            )
            
            # Log basic statistics
            logger.info(
                f"Data preview - First row sample: {df.iloc[0].to_dict() if len(df) > 0 else 'No data'}",
                extra={'request_id': self.request_id}
            )
            
            return df
            
        except pd.errors.EmptyDataError:
            logger.error(
                f"CSV file is empty: {filepath}",
                extra={'request_id': self.request_id}
            )
            raise
        except pd.errors.ParserError as e:
            logger.error(
                f"CSV parsing error: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error loading CSV: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def ingest(self, csv_path: str, skip_quality_check: bool = False) -> Dict:
        """
        Main ingestion pipeline with comprehensive logging and metrics
        
        Args:
            csv_path: Path to CSV file
            skip_quality_check: Whether to skip quality checks (for testing)
            
        Returns:
            dict: Ingestion results with metrics
            
        Raises:
            DataQualityError: If quality checks fail
            Exception: For other errors
        """
        self.metrics.start_time = datetime.now()
        
        logger.info(
            "=" * 80,
            extra={'request_id': self.request_id}
        )
        logger.info(
            "INGESTION PIPELINE STARTED",
            extra={'request_id': self.request_id}
        )
        logger.info(
            f"Request ID: {self.request_id}",
            extra={'request_id': self.request_id}
        )
        logger.info(
            f"Source file: {os.path.abspath(csv_path)}",
            extra={'request_id': self.request_id}
        )
        logger.info(
            f"Target database: {sanitize_db_url(self.db_url)}",
            extra={'request_id': self.request_id}
        )
        logger.info(
            f"Quality checks: {'DISABLED' if skip_quality_check else 'ENABLED'}",
            extra={'request_id': self.request_id}
        )
        logger.info(
            "=" * 80,
            extra={'request_id': self.request_id}
        )
        
        if skip_quality_check:
            logger.warning(
                "⚠️  Quality checks are SKIPPED - this should only be used for testing",
                extra={'request_id': self.request_id}
            )
        
        try:
            # Step 1: Load CSV
            logger.info(
                "STEP 1/4: Loading CSV file...",
                extra={'request_id': self.request_id}
            )
            raw_df = self.load_csv(csv_path)
            logger.info(
                "✓ STEP 1/4 COMPLETE",
                extra={'request_id': self.request_id}
            )
            
            # Step 2: Transform to schema
            logger.info(
                "STEP 2/4: Transforming data to target schema...",
                extra={'request_id': self.request_id}
            )
            billing_df, resources_df, parse_errors, rows_skipped = transform_to_schema(
                raw_df, 
                self.request_id
            )
            
            # Update metrics from transformation
            self.metrics.rows_transformed = len(billing_df)
            self.metrics.parse_errors = parse_errors
            self.metrics.rows_skipped = rows_skipped
            
            logger.info(
                "✓ STEP 2/4 COMPLETE",
                extra={'request_id': self.request_id}
            )
            
            # Step 3: Run quality checks
            quality_results = None
            if not skip_quality_check:
                logger.info(
                    "STEP 3/4: Running quality checks...",
                    extra={'request_id': self.request_id}
                )
                quality_results = run_quality_checks(
                    billing_df, 
                    resources_df, 
                    self.request_id
                )
                
                # Update metrics from quality checks
                self.metrics.quality_check_failures = quality_results['summary'].get('quality_check_failures', 0)
                
                if not quality_results['passed']:
                    error_checks = [c for c in quality_results['checks'] if c['severity'] == 'error' and not c['passed']]
                    error_messages = [c['message'] for c in error_checks]
                    
                    logger.error(
                        f"Quality checks FAILED with {len(error_checks)} critical errors",
                        extra={'request_id': self.request_id}
                    )
                    
                    for error_check in error_checks:
                        logger.error(
                            f"  - {error_check['message']}",
                            extra={'request_id': self.request_id}
                        )
                    
                    raise DataQualityError(
                        f"Quality checks failed with {len(error_checks)} critical error(s): {'; '.join(error_messages)}"
                    )
                
                logger.info(
                    "✓ STEP 3/4 COMPLETE - All quality checks passed",
                    extra={'request_id': self.request_id}
                )
            else:
                logger.info(
                    "⊘ STEP 3/4 SKIPPED - Quality checks disabled",
                    extra={'request_id': self.request_id}
                )
            
            # Step 4: Load to database
            logger.info(
                "STEP 4/4: Loading data to database...",
                extra={'request_id': self.request_id}
            )
            billing_loaded, resources_loaded = load_to_database(
                billing_df, 
                resources_df, 
                self.engine,
                self.request_id
            )
            
            # Update metrics from database load
            self.metrics.billing_records_loaded = billing_loaded
            self.metrics.resource_records_loaded = resources_loaded
            
            logger.info(
                "✓ STEP 4/4 COMPLETE",
                extra={'request_id': self.request_id}
            )
            
            # Finalize metrics
            self.metrics.end_time = datetime.now()
            duration = self.metrics.duration()
            
            # Prepare result
            result = {
                'status': 'success',
                'request_id': self.request_id,
                'metrics': self.metrics.to_dict(),
                'quality_checks': quality_results
            }
            
            # Log final summary
            logger.info(
                "=" * 80,
                extra={'request_id': self.request_id}
            )
            logger.info(
                "✓ INGESTION PIPELINE COMPLETED SUCCESSFULLY",
                extra={'request_id': self.request_id}
            )
            logger.info(
                f"Duration: {duration:.2f} seconds",
                extra={'request_id': self.request_id}
            )
            logger.info(
                f"Records processed: {self.metrics.rows_read} rows read → {self.metrics.rows_transformed} transformed",
                extra={'request_id': self.request_id}
            )
            logger.info(
                f"Records loaded: {self.metrics.billing_records_loaded} billing + {self.metrics.resource_records_loaded} resources",
                extra={'request_id': self.request_id}
            )
            logger.info(
                f"Errors encountered: {self.metrics.parse_errors} parse errors, {self.metrics.quality_check_failures} quality check failures",
                extra={'request_id': self.request_id}
            )
            logger.info(
                f"Database location: {sanitize_db_url(self.db_url)}",
                extra={'request_id': self.request_id}
            )
            logger.info(
                "=" * 80,
                extra={'request_id': self.request_id}
            )
            
            return result
            
        except DataQualityError as e:
            self.metrics.end_time = datetime.now()
            duration = self.metrics.duration()
            
            logger.error(
                "=" * 80,
                extra={'request_id': self.request_id}
            )
            logger.error(
                f"❌ INGESTION PIPELINE FAILED - Data Quality Error",
                extra={'request_id': self.request_id}
            )
            logger.error(
                f"Duration: {duration:.2f} seconds",
                extra={'request_id': self.request_id}
            )
            logger.error(
                f"Error: {str(e)}",
                extra={'request_id': self.request_id}
            )
            logger.error(
                "=" * 80,
                extra={'request_id': self.request_id}
            )
            
            raise
            
        except Exception as e:
            self.metrics.end_time = datetime.now()
            duration = self.metrics.duration()
            
            logger.error(
                "=" * 80,
                extra={'request_id': self.request_id}
            )
            logger.error(
                f"❌ INGESTION PIPELINE FAILED - Unexpected Error",
                extra={'request_id': self.request_id}
            )
            logger.error(
                f"Duration: {duration:.2f} seconds",
                extra={'request_id': self.request_id}
            )
            logger.error(
                f"Error type: {type(e).__name__}",
                extra={'request_id': self.request_id}
            )
            logger.error(
                f"Error message: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            logger.error(
                "=" * 80,
                extra={'request_id': self.request_id}
            )
            
            raise


def main():
    """CLI entry point for ingestion with environment configuration"""
    import os
    import uuid
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    db_url = os.getenv('DATABASE_URL', 'sqlite:///./data/cost_analytics.db')
    csv_path = os.getenv('BILLING_CSV_PATH', 'data/raw/billing_raw.csv')
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    # Generate request ID
    request_id = f"ingest-{uuid.uuid4().hex[:8]}"
    
    logger.info(
        f"Starting ingestion from CLI",
        extra={'request_id': request_id}
    )
    logger.info(
        f"Configuration: DB={db_url}, CSV={csv_path}, LOG_LEVEL={log_level}",
        extra={'request_id': request_id}
    )
    
    try:
        # Create loader and run ingestion
        loader = BillingDataLoader(db_url, request_id=request_id)
        result = loader.ingest(csv_path)
        
        # Print summary to console
        print("\n" + "=" * 80)
        print("✓ INGESTION SUCCESSFUL")
        print("=" * 80)
        print(f"Request ID: {result['request_id']}")
        print(f"Duration: {result['metrics']['duration_seconds']:.2f}s")
        print(f"Billing records: {result['metrics']['billing_records_loaded']}")
        print(f"Resource records: {result['metrics']['resource_records_loaded']}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ INGESTION FAILED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("=" * 80)
        
        return 1


if __name__ == '__main__':
    exit(main())