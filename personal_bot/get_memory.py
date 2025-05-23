"""
Bot Memory

This module implements the memory management system for the chat agent.
It maintains the conversation history and context.

Key functionalities:
- Stores and retrieves conversation history
- Maintains context across multiple interactions
- Manages memory persistence in session state
"""

from langchain.memory import ConversationBufferMemory


class BotMemory:
    """
    Class for managing the bot's memory.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BotMemory, cls).__new__(cls)
            cls._instance.memory = ConversationBufferMemory()
        return cls._instance

    def get_memory(self):
        return self.memory