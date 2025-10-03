"""
Database Operations Module
Handles loading transformed data into the database
"""

import pandas as pd
import logging
from datetime import datetime
from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)


def sanitize_db_url(db_url: str) -> str:
    """
    Remove sensitive information from database URL for logging
    
    Args:
        db_url: Database connection URL
        
    Returns:
        str: Sanitized URL safe for logging
    """
    if '://' in db_url:
        protocol, rest = db_url.split('://', 1)
        if '@' in rest:
            # Remove credentials
            _, host_part = rest.split('@', 1)
            return f"{protocol}://***@{host_part}"
    return db_url


def load_to_database(
    billing_df: pd.DataFrame, 
    resources_df: pd.DataFrame, 
    engine, 
    request_id: str = "default"
) -> tuple:
    """
    Load transformed data into database with detailed logging
    
    Args:
        billing_df: Billing dataframe
        resources_df: Resources dataframe
        engine: SQLAlchemy engine
        request_id: Request ID for logging
        
    Returns:
        tuple: (billing_records_loaded, resource_records_loaded)
    """
    logger.info(
        "=" * 60,
        extra={'request_id': request_id}
    )
    logger.info(
        "STARTING DATABASE LOAD",
        extra={'request_id': request_id}
    )
    logger.info(
        "=" * 60,
        extra={'request_id': request_id}
    )
    
    load_start = datetime.now()
    
    try:
        with engine.connect() as connection:
            # Check existing table state
            inspector = inspect(engine)
            
            # Load billing data
            logger.info(
                f"Loading {len(billing_df)} billing records to database...",
                extra={'request_id': request_id}
            )
            
            billing_exists = 'billing' in inspector.get_table_names()
            if billing_exists:
                # Get current row count
                result = connection.execute(text("SELECT COUNT(*) FROM billing"))
                existing_count = result.scalar_one()
                logger.info(
                    f"Billing table exists with {existing_count} existing records. Appending new data...",
                    extra={'request_id': request_id}
                )
            else:
                logger.info(
                    "Billing table does not exist. Creating new table...",
                    extra={'request_id': request_id}
                )
            
            billing_start = datetime.now()
            billing_df.to_sql('billing', connection, if_exists='append', index=False)
            connection.commit()  # Explicit commit for pandas to_sql
            billing_duration = (datetime.now() - billing_start).total_seconds()

            billing_records_loaded = len(billing_df)

            logger.info(
                f"✓ Successfully loaded {len(billing_df)} billing records in {billing_duration:.2f}s",
                extra={'request_id': request_id}
            )
            
            # Load resources data
            logger.info(
                f"Loading {len(resources_df)} resource records to database...",
                extra={'request_id': request_id}
            )
            
            resources_exists = 'resources' in inspector.get_table_names()
            if resources_exists:
                result = connection.execute(text("SELECT COUNT(*) FROM resources"))
                existing_count = result.scalar_one()
                logger.info(
                    f"Resources table exists with {existing_count} existing records. Appending new data...",
                    extra={'request_id': request_id}
                )
            else:
                logger.info(
                    "Resources table does not exist. Creating new table...",
                    extra={'request_id': request_id}
                )
            
            resources_start = datetime.now()
            resources_df.to_sql('resources', connection, if_exists='append', index=False)
            connection.commit()  # Explicit commit for pandas to_sql
            resources_duration = (datetime.now() - resources_start).total_seconds()

            resource_records_loaded = len(resources_df)

            logger.info(
                f"✓ Successfully loaded {len(resources_df)} resource records in {resources_duration:.2f}s",
                extra={'request_id': request_id}
            )
            
            # Verify load
            logger.info(
                "Verifying database load...",
                extra={'request_id': request_id}
            )
            
            result = connection.execute(text("SELECT COUNT(*) FROM billing"))
            total_billing = result.scalar_one()
            
            result = connection.execute(text("SELECT COUNT(*) FROM resources"))
            total_resources = result.scalar_one()
        
        logger.info(
            f"Verification complete - Total records in database: {total_billing} billing, {total_resources} resources",
            extra={'request_id': request_id}
        )
        
        load_duration = (datetime.now() - load_start).total_seconds()
        
        logger.info(
            "=" * 60,
            extra={'request_id': request_id}
        )
        logger.info(
            f"DATABASE LOAD COMPLETE in {load_duration:.2f}s",
            extra={'request_id': request_id}
        )
        logger.info(
            "=" * 60,
            extra={'request_id': request_id}
        )
        
        return billing_records_loaded, resource_records_loaded
        
    except Exception as e:
        logger.error(
            f"Database load failed: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise