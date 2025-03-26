"""
Utils Module

This module provides utility functions for the VPC log chatbot.
"""

import logging
import os
from typing import Dict, Any, List


def setup_logging(log_file: str = "vpc_chatbot.log") -> None:
    """
    Set up logging for the application.

    Args:
        log_file: Path to the log file
    """
    # Ensure the directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that the specified directory exists.

    Args:
        directory_path: Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def format_log_entry(log_entry: Dict[str, Any]) -> str:
    """
    Format a log entry for display.

    Args:
        log_entry: Log entry to format

    Returns:
        Formatted log entry
    """
    formatted = []

    # Add timestamp if available
    if 'timestamp' in log_entry:
        formatted.append(f"Time: {log_entry['timestamp']}")

    # Add level if available
    if 'level' in log_entry:
        formatted.append(f"Level: {log_entry['level']}")

    # Add source if available
    if 'source' in log_entry:
        formatted.append(f"Source: {log_entry['source']}")

    # Add message if available
    if 'message' in log_entry:
        formatted.append(f"Message: {log_entry['message']}")

    # Add any other fields
    for key, value in log_entry.items():
        if key not in ['timestamp', 'level', 'source', 'message']:
            formatted.append(f"{key}: {value}")

    return '\n'.join(formatted)


def batch_logs(logs: List[Dict[str, Any]], batch_size: int = 5) -> List[List[Dict[str, Any]]]:
    """
    Batch logs into groups.

    Args:
        logs: List of log entries
        batch_size: Size of each batch

    Returns:
        List of batches of log entries
    """
    return [logs[i:i + batch_size] for i in range(0, len(logs), batch_size)]