"""
Main Module

Entry point for the VPC Log Chatbot application.
"""

import logging
import os
from utils import setup_logging
from streamlit_app import StreamlitApp


def main(vpc_chatbot=None):
    """Main entry point for the application."""
    # Set up logging
    log_dir = r"C:\Users\adity\PycharmProjects\LOG_Chatbot"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "vpc_chatbot.log")
    setup_logging(log_file)

    # Create and run Streamlit app
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()