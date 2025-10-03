"""
Tag Parsing Utilities Module
Handles parsing and extraction of tags from billing data
"""

import json
import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


def parse_tags(tags_str: str, row_index: Optional[int] = None, request_id: str = "default") -> dict:
    """
    Parse tags string into dictionary with error tracking
    
    Args:
        tags_str: JSON string of tags
        row_index: Row index for logging purposes
        request_id: Request ID for logging
        
    Returns:
        dict: Parsed tags or empty dict if parsing fails
    """
    if pd.isna(tags_str) or tags_str == 'NULL' or tags_str == '':
        return {}
    
    try:
        if isinstance(tags_str, str):
            parsed = json.loads(tags_str)
            return parsed if isinstance(parsed, dict) else {}
        return {}
    except json.JSONDecodeError as e:
        logger.warning(
            f"Failed to parse tags at row {row_index}: {tags_str[:100]}... Error: {str(e)}",
            extra={'request_id': request_id}
        )
        return {}
    except Exception as e:
        logger.warning(
            f"Unexpected error parsing tags at row {row_index}: {str(e)}",
            extra={'request_id': request_id}
        )
        return {}


def extract_from_tags(tags: dict, key: str) -> Optional[str]:
    """
    Extract value from tags dictionary with case-insensitive key matching
    
    Args:
        tags: Dictionary of tags
        key: Key to extract (case-insensitive)
        
    Returns:
        str: Value or None
    """
    if not tags:
        return None
    
    # Try exact match first
    if key in tags:
        return tags[key]
    
    # Try case-insensitive match
    for tag_key, tag_value in tags.items():
        if tag_key.lower() == key.lower():
            return tag_value
    
    # Handle common variations
    key_variations = {
        'environment': ['env', 'stage', 'environ', 'deployment_env'],
        'owner': ['owned_by', 'team', 'contact', 'owner_email'],
        'application': ['app', 'application_name', 'app_name', 'service']
    }
    
    if key in key_variations:
        for variation in key_variations[key]:
            if variation in tags:
                return tags[variation]
    
    return None