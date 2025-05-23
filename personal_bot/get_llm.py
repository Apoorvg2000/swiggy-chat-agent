"""
LLM Configuration

This module handles the configuration and initialization of the Language Model.
It manages API keys and creates the LLM instance for use across the application.

Key functionalities:
- Loads and validates API keys
- Configures LLM parameters
- Creates and returns LLM instance
- Manages API key security
"""

from langchain_groq import ChatGroq
import os
import httpx
from dotenv import load_dotenv
import logging

httpx_logger = logging.getLogger("httpx")

httpx_logger.setLevel(logging.ERROR)

load_dotenv("../personal_bot/.env")

groq_api_key = os.getenv("GROQ_API_KEY")

def get_llm(model_name="llama3-70b-8192", temperature=0.5, stop_words=None, max_tokens=512):
    model = ChatGroq(
        model_name=model_name,
        temperature=temperature,
        stop=stop_words,
        max_tokens=max_tokens,
        api_key=groq_api_key,
        http_client=httpx.Client(verify=False)
    )
    
    return model