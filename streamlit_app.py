
"""
Streamlit App Module

This module provides the Streamlit web interface for the VPC log chatbot.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import logging
import traceback
import time
import os
from pathlib import Path

from log_processor import VPCLogProcessor
from llm_integration import LangChainLLM
from chatbot_core import VPCLogChatbot

logger = logging.getLogger(__name__)


class StreamlitApp:
    """Streamlit application for the VPC log chatbot."""

    def __init__(self):
        """Initialize the Streamlit application."""
        self.log_processor = None
        self.llm = None
        self.chatbot = None

        # Add initialization flag to session state
        if "initialized" not in st.session_state:
            st.session_state.initialized = False
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Initialize configuration parameters - with option to override via UI
        self._init_config_state()

    def _init_config_state(self):
        """Initialize configuration parameters in session state."""
        # Set default log directory
        if "log_directory" not in st.session_state:
            st.session_state.log_directory = r"C:\Users\adity\PycharmProjects\LOG_Chatbot\vpc_logs_large.log"

        # Set default model
        if "model_name" not in st.session_state:
            st.session_state.model_name = "deepseek-r1:1.5b"

        # Set default API URL
        if "api_url" not in st.session_state:
            st.session_state.api_url = "http://localhost:11434"

    def run(self):
        """Run the Streamlit application."""
        st.set_page_config(page_title="VPC Log Chatbot", layout="wide")

        # Sidebar
        with st.sidebar:
            st.title("VPC Log Chatbot")
            st.markdown("### Configuration")

            # Allow users to set configuration via UI
            new_log_directory = st.text_input("Log Directory/File Path", value=st.session_state.log_directory)
            new_model_name = st.text_input("Ollama Model", value=st.session_state.model_name)
            new_api_url = st.text_input("Ollama API URL", value=st.session_state.api_url)

            # Update session state if values changed
            if (new_log_directory != st.session_state.log_directory or
                    new_model_name != st.session_state.model_name or
                    new_api_url != st.session_state.api_url):
                st.session_state.log_directory = new_log_directory
                st.session_state.model_name = new_model_name
                st.session_state.api_url = new_api_url
                # Reset initialization if config changed
                st.session_state.initialized = False

            # Display initialization status
            init_status = "✅ Initialized" if st.session_state.initialized else "❌ Not Initialized"
            st.info(init_status)

            # Initialize button
            if st.button("Initialize/Reinitialize"):
                self._initialize()

            st.markdown("---")
            st.markdown("### Navigation")
            page = st.radio("Select page", ["Chatbot", "Dashboard"])

        # Main content
        if page == "Chatbot":
            self._render_chatbot_page()
        else:
            self._render_dashboard_page()

    def _initialize(self):
        """
        Initialize the chatbot components with improved error handling.
        """
        try:
            # Reset initialization state
            st.session_state.initialized = False

            with st.sidebar:
                progress_text = "Initializing..."
                progress_bar = st.progress(0, text=progress_text)

                # Step 1: Initialize log processor
                progress_bar.progress(10, text="Creating log processor...")
                st.info(f"Starting initialization with path: {st.session_state.log_directory}")
                self.log_processor = VPCLogProcessor(st.session_state.log_directory)

                # Store in session state
                st.session_state.log_processor = self.log_processor

                # Step 2: Load logs
                progress_bar.progress(30, text="Loading logs...")
                success = self.log_processor.load_logs()
                if not success:
                    st.error("Failed to load logs from the specified path.")
                    return

                # Step 3: Initialize LLM
                progress_bar.progress(50, text="Creating LLM...")
                try:
                    self.llm = LangChainLLM(st.session_state.model_name, st.session_state.api_url)
                    # Store in session state
                    st.session_state.llm = self.llm
                except Exception as llm_error:
                    st.error(f"Failed to initialize LLM: {str(llm_error)}")
                    st.error("Make sure Ollama is running and the model is available.")
                    return

                # Step 4: Initialize chatbot
                progress_bar.progress(80, text="Creating chatbot...")
                try:
                    self.chatbot = VPCLogChatbot(self.log_processor, self.llm)
                    # Store in session state
                    st.session_state.chatbot = self.chatbot
                except Exception as chatbot_error:
                    st.error(f"Failed to initialize chatbot: {str(chatbot_error)}")
                    return

                # Step 5: Set initialization flag
                progress_bar.progress(100, text="Initialization complete!")
                st.session_state.initialized = True

                # Success message
                st.success(f"Initialization successful! Loaded {len(self.log_processor.logs)} log entries.")

        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            st.error(f"Exception details: {traceback.format_exc()}")

    def _render_chatbot_page(self):
        """Render the chatbot page with improved user experience."""
        st.title("VPC Log Chatbot")

        # Add instruction above chat
        st.write("Ask questions about your VPC logs or general log analysis questions.")

        # Check initialization
        if not st.session_state.initialized:
            st.warning("System is not initialized. Please initialize using the button in the sidebar.")
            return

        # Get chatbot instance from session state
        if "chatbot" not in st.session_state or st.session_state.chatbot is None:
            st.warning("System is not fully initialized. Click 'Initialize/Reinitialize' in the sidebar.")
            return

        self.chatbot = st.session_state.chatbot

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input with placeholder text
        if prompt := st.chat_input("Ask about VPC logs, security events, connection patterns..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response with streaming effect
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Processing your query..."):
                        response = self.chatbot.process_query(prompt)

                    # Display response
                    st.markdown(response)

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    def _render_dashboard_page(self):
        """Render the dashboard page."""
        st.title("VPC Log Chatbot Dashboard")

        # Check initialization
        if not st.session_state.initialized:
            st.warning("System is not initialized. Please initialize using the button in the sidebar.")
            return

        # Get components from session state
        try:
            self.chatbot = st.session_state.chatbot
            self.log_processor = st.session_state.log_processor
            self.llm = st.session_state.llm
        except (KeyError, AttributeError):
            st.error("One or more components are missing. Please initialize using the button in the sidebar.")
            return

        # Create layout
        col1, col2 = st.columns(2)

        # Log summary
        with col1:
            st.subheader("Log Summary")
            try:
                log_summary = self.log_processor.get_log_summary()

                if "error" in log_summary:
                    st.error(log_summary["error"])
                else:
                    st.metric("Total Logs", log_summary["total_logs"])
                    st.metric("Error Count", log_summary["error_count"])

                    if log_summary["source_distribution"]:
                        st.subheader("Source Distribution")
                        source_df = pd.DataFrame({
                            "Source": log_summary["source_distribution"].keys(),
                            "Count": log_summary["source_distribution"].values()
                        })
                        fig = px.bar(source_df, x="Source", y="Count")
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying log summary: {str(e)}")

        # LLM metrics
        with col2:
            st.subheader("LLM Metrics")
            try:
                llm_metrics = self.llm.get_metrics()

                st.metric("Total Requests", llm_metrics["total_requests"])
                st.metric("Total Tokens", llm_metrics["total_tokens"])
                st.metric("Avg Response Time (s)", round(llm_metrics["average_response_time"], 2))
            except Exception as e:
                st.error(f"Error displaying LLM metrics: {str(e)}")

        # Usage statistics
        st.subheader("Usage Statistics")
        try:
            usage_stats = self.chatbot.get_usage_stats()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Queries", usage_stats["total_queries"])
            with col2:
                st.metric("Queries in Last Hour", usage_stats["queries_per_hour"])
            with col3:
                st.metric("Avg Response Time (s)", round(usage_stats["average_response_time"], 2))

            # Recent queries
            st.subheader("Recent Queries")
            if usage_stats["recent_queries"]:
                recent_queries_df = pd.DataFrame(usage_stats["recent_queries"], columns=["Query", "Timestamp"])
                st.dataframe(recent_queries_df, use_container_width=True)
            else:
                st.info("No queries yet.")

            # Response time chart
            if self.chatbot.usage_stats["response_times"]:
                st.subheader("Response Time Trend")
                response_time_df = pd.DataFrame({
                    "Timestamp": self.chatbot.usage_stats["timestamps"],
                    "Response Time (s)": self.chatbot.usage_stats["response_times"]
                })
                fig = px.line(response_time_df, x="Timestamp", y="Response Time (s)")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying usage statistics: {str(e)}")