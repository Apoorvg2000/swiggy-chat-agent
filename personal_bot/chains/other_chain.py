"""
Web Search Chain

This module implements a chain for handling general queries.
It processes queries that don't fit into specific intents and extracts key phrases and words
are then used to do a web search.

Key functionalities:
- Handles general knowledge queries
- Extracts key phrases and words from the query
"""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import sys
sys.path.append("./personal_bot")
from ..get_llm import get_llm
from ..get_memory import BotMemory

def other_chain():
    llm = get_llm("llama-3.3-70b-versatile", temperature=0.3)
    memory = BotMemory().get_memory()

    other_chain_prompt = PromptTemplate(
        input_variables=["query"],
        template="""Instructions:
You are an expert assistant designed to convert natural language user queries into highly effective web search strings. Your task is to analyze the user's query and generate a single, well-formed search phrase that captures the key details and intent of the user’s question or request.

This search phrase should be:

1. Concise but complete (DO NOT missing critical information).
2. Optimized for a web search engine (what a human would type to get good results).
3. Written in natural language or keyword-style, depending on which is most effective for the query.
4. Include locations, timeframes, product names, or specific actions if relevant.
5. Exclude filler words or context that won’t help the search engine.
6. Do not add any explanation or extra text, output only valid JSON.

Output your response strictly in this JSON format:

{{
  "response": "<web_search_string>"
}}

Examples:

Example 1:
User: "Looking for budget-friendly hotels in Manali for 2 people in June"

Response:
{{
  "response": "budget hotels in Manali for 2 people in June"
}}

Example 2:
User: "Can you suggest some good Italian restaurants near Andheri West?"

Response:
{{
  "response": "best Italian restaurants near Andheri West"
}}

Example 3:
User: "How to fix a leaking tap in the kitchen?"

Response:
{{
  "response": "how to fix leaking kitchen tap"
}}

Example 4:
User: "I need a gift idea for my mom’s birthday under 1000 rupees"

Response:
{{
  "response": "gift ideas for mom birthday under 1000 rupees"
}}


Strictly follow the above instructions and examples and output only the JSON response.

User: {query}

"""
    )

    other_chain = LLMChain(
        llm=llm,
        prompt=other_chain_prompt,
        memory=memory
    )

    return other_chain