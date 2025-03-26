
"""
LLM Integration Module

This module provides integration with LangChain and Ollama for LLM capabilities.
"""

import time
import logging
from typing import Dict, Any, List
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)


class LangChainLLM:
    """Integration with LangChain and Ollama LLM service."""

    def __init__(self, model_name: str = "deepseek-r1:1.5b", api_url: str = "http://localhost:11434"):
        """
        Initialize the LangChain LLM integration.

        Args:
            model_name: Name of the Ollama model to use
            api_url: URL of the Ollama API
        """
        self.model_name = model_name
        self.api_url = api_url

        # Initialize metrics
        self.metrics = {
            "requests": 0,
            "tokens_used": 0,
            "response_times": []
        }

        # Initialize LangChain components with optimized parameters
        self.llm = OllamaLLM(
            model=self.model_name,
            base_url=self.api_url,
            temperature=0.3,  # Lower temperature for more focused responses
            max_tokens=512,  # Limit response length
            request_timeout=30.0  # Set timeout
        )

        # Define prompt template with all required inputs
        template = """
        You are an intelligent Log Analysis Assistant, skilled in analyzing VPC logs.

        ### User Query:
        {query}

        ### Available Log Data:
        {context}

        ### Previous Conversation:
        {chat_history}

        ### Instructions:
        1. Answer directly based on the log data when available
        2. For log filtering questions, analyze the relevant entries
        3. Be concise and focused
        4. If no logs match, state that clearly

        ### Answer:
        """

        self.prompt = PromptTemplate(
            input_variables=["query", "context", "chat_history"],
            template=template
        )

        # Create memory
        self.memory = ConversationBufferMemory(memory_key="chat_history")

        # Create LLM chain with all components
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=True
        )


    def get_response(self, query: str, context: str = None, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Get a response from the LLM using LangChain.

        Args:
            query: User query
            context: Additional context to provide to the LLM
            chat_history: Chat history for conversation context

        Returns:
            LLM response
        """
        start_time = time.time()

        # Handle casual greetings and small talk efficiently
        greetings = ["hi", "hello", "hey", "hii", "good morning", "good evening"]
        small_talk = ["how are you", "what's up", "how is your day", "how have you been"]

        query_lower = query.lower().strip()

        if query_lower in greetings:
            return "Hello! 😊 How can I assist you today?"

        if any(phrase in query_lower for phrase in small_talk):
            return "I'm here to help! How can I assist you with log analysis or troubleshooting?"

        # Ensure context is always a string
        if not context or context.strip() == "No matching logs found.":
            context = "No logs matched your specific query."

        # Format chat history for the prompt
        formatted_history = ""
        if chat_history:
            for message in chat_history[-4:]:  # Last 4 messages for context
                formatted_history += f"{message['role'].capitalize()}: {message['content']}\n"

        # Detect if this is a follow-up question
        is_followup = any(word in query_lower for word in
                          ['again', 'previous', 'before', 'those', 'them', 'it', 'that'])

        try:
            logger.info("Processing query: %s (follow-up: %s)", query, is_followup)

            # Invoke the chain with all necessary inputs
            response = self.chain.invoke({
                "query": query,
                "context": context,
                "chat_history": formatted_history
            })

            # Extract response text based on response format
            if isinstance(response, dict):
                if "answer" in response:
                    response_text = response["answer"]
                elif "text" in response:
                    response_text = response["text"]
                else:
                    response_text = str(response)
            else:
                response_text = str(response)

            # Clean up response if needed
            if "<think>" in response_text:
                response_parts = response_text.split("</think>")
                if len(response_parts) > 1:
                    response_text = response_parts[1].strip()

            # Update metrics
            self.metrics["requests"] += 1
            self.metrics["response_times"].append(time.time() - start_time)

            return response_text

        except Exception as e:
            logger.error(f"Error calling LLM via LangChain: {str(e)}")
            logger.error(traceback.format_exc())  # Add full traceback for debugging
            return f"Error: {str(e)}"


    def get_metrics(self) -> Dict[str, Any]:
        """
        Get usage metrics for the LLM.

        Returns:
            Dictionary containing metrics
        """
        if not self.metrics["response_times"]:
            avg_response_time = 0
        else:
            avg_response_time = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])

        return {
            "total_requests": self.metrics["requests"],
            "total_tokens": self.metrics["tokens_used"],
            "average_response_time": avg_response_time
        }