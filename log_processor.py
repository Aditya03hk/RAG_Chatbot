
"""
VPC Log Processor Module

This module provides functionality for processing and analyzing VPC logs.
"""

import os
import json
import logging
import pandas as pd
from typing import List, Dict, Any
import datetime

logger = logging.getLogger(__name__)


class VPCLogProcessor:
    """Processes VPC logs for querying and analysis."""

    def __init__(self, log_directory: str):
        """
        Initialize the VPC log processor.

        Args:
            log_directory: Directory or file path containing VPC logs
        """
        self.log_directory = log_directory
        self.logs = []
        self.log_df = None
        self.log_cache = {}  # Add cache for repeated queries

    def load_logs(self):
        """Load logs from the specified directory or file."""
        logger.info(f"Loading logs from {self.log_directory}")

        try:
            all_logs = []

            # Check if the path is a directory
            if os.path.isdir(self.log_directory):
                # Original directory handling code
                for filename in os.listdir(self.log_directory):
                    full_path = os.path.join(self.log_directory, filename)
                    if os.path.isfile(full_path):
                        logs = self._process_file(full_path)
                        all_logs.extend(logs)
            # Check if the path is a file
            elif os.path.isfile(self.log_directory):
                logs = self._process_file(self.log_directory)
                all_logs.extend(logs)
            else:
                logger.error(f"Path does not exist: {self.log_directory}")
                return False

            self.logs = all_logs

            # Create DataFrame only if there are logs
            if all_logs:
                try:
                    self.log_df = pd.DataFrame(all_logs)
                except Exception as df_error:
                    logger.error(f"Error creating DataFrame: {str(df_error)}")
                    # Create a more compatible DataFrame structure
                    self.log_df = pd.DataFrame([{"message": str(log)} for log in all_logs])

            logger.info(f"Loaded {len(all_logs)} log entries")
            return True

        except Exception as e:
            logger.error(f"Error loading logs: {str(e)}")
            return False

    def _process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single log file."""
        logs = []
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        try:
                            log_entries = json.loads(content)
                            if isinstance(log_entries, list):
                                logs.extend(log_entries)
                            else:
                                logs.append(log_entries)
                        except json.JSONDecodeError:
                            # Try line by line if the whole file isn't valid JSON
                            with open(file_path, 'r', encoding='utf-8') as f2:
                                for line in f2:
                                    if line.strip():
                                        try:
                                            log_entry = json.loads(line.strip())
                                            logs.append(log_entry)
                                        except json.JSONDecodeError:
                                            logs.append({"message": line.strip(),
                                                         "timestamp": datetime.datetime.now().isoformat()})
            elif file_ext in ['.log', '.txt']:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        if line.strip():
                            try:
                                log_entry = json.loads(line.strip())
                                logs.append(log_entry)
                            except json.JSONDecodeError:
                                # Handle non-JSON log formats
                                logs.append({"message": line.strip(), "timestamp": datetime.datetime.now().isoformat()})
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")

        return logs

    def filter_logs(self, query: str) -> List[Dict[str, Any]]:
        """
        Filter logs based on a query string with optimized performance.

        Args:
            query: Search query

        Returns:
            List of matching log entries
        """
        if not self.logs:
            logger.warning("No logs loaded")
            return []

        # Check cache first
        if query in self.log_cache:
            logger.info(f"Using cached results for query: {query}")
            return self.log_cache[query]

        # Convert query to lowercase and split into terms
        query_terms = query.lower().split()

        # If very generic query, return sample
        if len(query_terms) <= 2 and all(len(term) < 4 for term in query_terms):
            return self.logs[:5]

        # Special handling for protocol queries
        protocol_search = "protocol" in query.lower()
        protocol_numbers = [term for term in query_terms if term.isdigit()]

        # Filter logs efficiently
        filtered_logs = []
        for log in self.logs:
            # Convert log to string for searching
            log_str = json.dumps(log).lower()

            # Special handling for protocol searches
            if protocol_search and protocol_numbers:
                for protocol in protocol_numbers:
                    if f'"protocol":{protocol}' in log_str or f'"protocol": {protocol}' in log_str:
                        filtered_logs.append(log)
                        break
                continue

            # Regular term matching - require all terms to be present
            if all(term in log_str for term in query_terms):
                filtered_logs.append(log)

            # Limit to reasonable number for performance
            if len(filtered_logs) >= 50:
                break

        # Limit the number of logs returned to avoid overloading the LLM
        result = filtered_logs[:10]  # Return at most 10 logs

        # Cache the results
        self.log_cache[query] = result

        # If no matches found but we have logs, return a few samples
        if not result and self.logs:
            logger.info(f"No specific matches found for query: {query}, returning sample logs")
            return self.logs[:3]

        return result

    def get_log_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the loaded logs.

        Returns:
            Dictionary containing summary statistics
        """
        if self.log_df is None or self.log_df.empty:
            return {"error": "No logs loaded"}

        summary = {
            "total_logs": len(self.log_df),
            "time_range": {
                "start": None,
                "end": None
            },
            "source_distribution": {},
            "error_count": 0
        }

        # Handle timestamp column if it exists
        if 'timestamp' in self.log_df.columns:
            try:
                summary["time_range"]["start"] = self.log_df['timestamp'].min()
                summary["time_range"]["end"] = self.log_df['timestamp'].max()
            except Exception:
                pass  # Skip if timestamps can't be processed

        # Handle source column if it exists
        if 'source' in self.log_df.columns:
            try:
                summary["source_distribution"] = self.log_df['source'].value_counts().to_dict()
            except Exception:
                pass  # Skip if source can't be processed

        # Handle level column if it exists
        if 'level' in self.log_df.columns:
            try:
                summary["error_count"] = self.log_df['level'].value_counts().get('ERROR', 0)
            except Exception:
                pass  # Skip if level can't be processed

        return summary