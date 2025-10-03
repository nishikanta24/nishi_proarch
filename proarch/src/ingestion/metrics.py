"""
Ingestion Metrics Module
Tracks metrics during the data ingestion process
"""

from datetime import datetime


class IngestionMetrics:
    """Track metrics during ingestion process"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.rows_read = 0
        self.rows_transformed = 0
        self.rows_skipped = 0
        self.billing_records_loaded = 0
        self.resource_records_loaded = 0
        self.parse_errors = 0
        self.quality_check_failures = 0
        
    def duration(self) -> float:
        """Calculate duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary"""
        return {
            'duration_seconds': self.duration(),
            'rows_read': self.rows_read,
            'rows_transformed': self.rows_transformed,
            'rows_skipped': self.rows_skipped,
            'billing_records_loaded': self.billing_records_loaded,
            'resource_records_loaded': self.resource_records_loaded,
            'parse_errors': self.parse_errors,
            'quality_check_failures': self.quality_check_failures
        }