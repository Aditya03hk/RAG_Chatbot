
"""
Chatbot Core Module

This module provides the core functionality for the VPC log chatbot.
"""

import json
import time
import datetime
import logging
from typing import Dict, Any, List

from log_processor import VPCLogProcessor
from llm_integration import LangChainLLM

logger = logging.getLogger(__name__)


class VPCLogChatbot:
    """Main chatbot application integrating log processing and LLM."""

    def __init__(self, log_processor: VPCLogProcessor, llm: LangChainLLM):
        """
        Initialize the VPC log chatbot.

        Args:
            log_processor: VPC log processor
            llm: LLM integration
        """
        self.log_processor = log_processor
        self.llm = llm
        self.usage_stats = {
            "queries": [],
            "timestamps": [],
            "response_times": []
        }
        self.chat_history = []
        self.last_query_time = 0

    def __init__(self, log_processor: VPCLogProcessor, llm: LangChainLLM):
        """
        Initialize the VPC log chatbot.

        Args:
            log_processor: VPC log processor
            llm: LLM integration
        """
        self.log_processor = log_processor
        self.llm = llm
        self.usage_stats = {
            "queries": [],
            "timestamps": [],
            "response_times": []
        }
        self.chat_history = []
        self.last_query_time = 0


    def process_query(self, query: str) -> str:
        """
        Process a user query with improved performance.

        Args:
            query: User query

        Returns:
            Chatbot response
        """
        start_time = time.time()

        try:
            # Check if logs are loaded
            if not self.log_processor.logs:
                return "No logs loaded. Please load logs first."

            # Implement rate limiting to prevent overloading the LLM
            time_since_last_query = time.time() - self.last_query_time
            if time_since_last_query < 1.0 and self.last_query_time > 0:
                time.sleep(1.0 - time_since_last_query)  # Brief pause to prevent rapid-fire queries

            self.last_query_time = time.time()

            # Filter logs based on the query
            filtered_logs = self.log_processor.filter_logs(query)

            # Create optimized context for the LLM
            if filtered_logs:
                # Limit to 3 logs for context to improve performance
                log_sample = filtered_logs[:3]
                context = json.dumps(log_sample, indent=2)
                context += f"\n\nTotal matching logs: {len(filtered_logs)}"

                # Truncate if too large (max 4000 chars)
                if len(context) > 4000:
                    context = context[:4000] + "... [truncated]"
            else:
                context = "No matching logs found."

            # Add query to chat history
            self.chat_history.append({"role": "user", "content": query})

            # Get response from LLM
            response = self.llm.get_response(query, context, self.chat_history)

            # Add response to chat history
            self.chat_history.append({"role": "assistant", "content": response})

            # Update usage stats
            response_time = time.time() - start_time
            self.usage_stats["queries"].append(query)
            self.usage_stats["timestamps"].append(datetime.datetime.now())
            self.usage_stats["response_times"].append(response_time)

            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error processing your query: {str(e)}"
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for the chatbot.

        Returns:
            Dictionary containing usage statistics
        """
        if not self.usage_stats["response_times"]:
            avg_response_time = 0
        else:
            avg_response_time = sum(self.usage_stats["response_times"]) / len(self.usage_stats["response_times"])

        return {
            "total_queries": len(self.usage_stats["queries"]),
            "average_response_time": avg_response_time,
            "queries_per_hour": len([t for t in self.usage_stats["timestamps"] if
                                     t > datetime.datetime.now() - datetime.timedelta(hours=1)]),
            "recent_queries": list(zip(self.usage_stats["queries"][-5:], self.usage_stats["timestamps"][-5:]))
        }