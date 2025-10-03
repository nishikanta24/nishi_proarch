#!/usr/bin/env python3
"""
Simple script to run data ingestion
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ingestion.loader import main as ingestion_main

if __name__ == '__main__':
    # Load environment variables
    load_dotenv()

    # Set logging to INFO level
    logging.basicConfig(level=logging.INFO)

    # Run ingestion
    exit_code = ingestion_main()
    sys.exit(exit_code)
