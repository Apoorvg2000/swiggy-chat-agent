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