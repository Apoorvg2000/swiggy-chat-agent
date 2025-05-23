"""
This module implements the main chat agent that handles user interactions and processes different types of intents.
The ChatAgent class orchestrates the flow of conversation, from understanding user queries to generating appropriate responses.
It uses various chains for intent classification, entity extraction, and follow-up question generation.
"""


from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import streamlit as st
import re
import json
import logging
import sys
sys.path.append("../")

from personal_bot.chains.intent_classifier_chain import intent_classifier_chain
from personal_bot.chains.contextual_chain import contextual_query_chain
from personal_bot.chains.extract_key_entities_chain import extract_key_entities_chain
from personal_bot.chains.followup_questions_chain import followup_questions_chain
from personal_bot.chains.other_chain import other_chain
from personal_bot.get_memory import BotMemory
from personal_bot.utils.intent_utils import DiningIntent, TravelIntent, GiftingIntent, CabIntent

class ChatAgent:
    """
    Main chat agent class that handles user interactions and processes different types of intents.
    
    This class manages the entire conversation flow, including:
    - Processing user queries with context
    - Classifying user intents
    - Extracting relevant entities
    - Generating follow-up questions
    - Handling web searches for general queries
    - Managing conversation memory
    """
    
    def __init__(self):
        """
        Initialize the ChatAgent with necessary components and logging setup.
        Sets up various chains for processing different aspects of the conversation.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.contextual_query_chain = contextual_query_chain()
        self.intent_classifier_chain = intent_classifier_chain()
        self.extract_key_entities_chain = extract_key_entities_chain()
        self.follow_up_questions_chain = followup_questions_chain()
        self.last_query = ""
        self.memory = BotMemory().get_memory()
        self.other_chain = other_chain()


    def get_contextual_query_response(self, query):
        """
        Process the user query with context from previous conversation.
        
        Args:
            query (str): The current user query
            
        Returns:
            str: The processed query with context resolved
        """
        
        last_query = f"User: {self.last_query.strip()}" if self.last_query.strip() not in ["", None] else ""
        if last_query:
            input = f"Context:\n{last_query}\n\nquery: {query}"
        else:
            input = f"Context:\n\nquery: {query}"

        self.last_query = query

        contextual_chain_response = self.contextual_query_chain.run({"input": input})

        match = re.search(r"\{.*\}", contextual_chain_response, re.DOTALL)
        if match:
            contextual_chain_response = match.group(0)

        try:
            contextual_chain_response = json.loads(contextual_chain_response)
        except Exception as e:
            self.logger.error(f"Error in contextual query chain: {e}")
            return "An error occurred while processing your query. Please try again."
        
        absolute_query = contextual_chain_response["response"]

        return absolute_query
    

    def get_intent_classification_response(self, absolute_query):
        """
        Classify the user's intent from their query.
        
        Args:
            absolute_query (str): The processed user query
            
        Returns:
            tuple: (intent_category, confidence_score)
        """

        intent_chain_response = self.intent_classifier_chain.run({"query": absolute_query})

        match = re.search(r"\{.*\}", intent_chain_response, re.DOTALL)
        if match:
            intent_chain_response = match.group(0)

        try:
            intent_chain_response = json.loads(intent_chain_response)
        except Exception as e:
            self.logger.error(f"Error in intent classification chain: {e}")
            return "An error occurred while processing your query. Please try again."
        
        intent_category = intent_chain_response["intent_category"]
        confidence_score = intent_chain_response["confidence_score"]

        return intent_category, confidence_score
    

    def get_extracted_entities_response(self, absolute_query, keys):
        """
        Extract relevant entities from the user query based on the intent type.
        
        Args:
            absolute_query (str): The processed user query
            keys (list): List of entity keys to extract
            
        Returns:
            dict: Extracted entities and their values
        """

        extract_keys_input = f"Key Entities: {keys}\nUser: {absolute_query}"
        entities_chain_response = self.extract_key_entities_chain.run({"input": extract_keys_input})

        match = re.search(r"\{.*\}", entities_chain_response, re.DOTALL)
        if match:
            entities_chain_response = match.group(0)

        try:
            entities_chain_response = json.loads(entities_chain_response)
        except Exception as e:
            self.logger.error(f"Error in extracting entities chain: {e}")
            return "An error occurred while processing your query. Please try again."
        
        return entities_chain_response
    

    def get_follow_up_questions(self, query, intent_entities):
        """
        Generate relevant follow-up questions based on the current query and extracted entities.
        
        Args:
            query (str): The user's query
            intent_entities (dict): The extracted entities for the current intent
            
        Returns:
            list: List of follow-up questions
        """

        input = f"User: {query}\n\nInfo:\n{json.dumps(intent_entities, indent=4)}"
        follow_up_questions_chain_response = self.follow_up_questions_chain.run({"input": input})

        match = re.search(r"\{.*\}", follow_up_questions_chain_response, re.DOTALL)
        if match:
            follow_up_questions_chain_response = match.group(0)

        try:
            follow_up_questions_chain_response = json.loads(follow_up_questions_chain_response)
            follow_up_questions = follow_up_questions_chain_response["response"]
        except Exception as e:
            self.logger.error(f"Error in follow up questions chain: {e}")
            return "An error occurred while processing your query. Please try again."
        
        return follow_up_questions
    

    def get_web_search_response(self, absolute_query):
        """
        Perform a web search for general queries that don't match specific intents.
        
        Args:
            absolute_query (str): The processed user query
            
        Returns:
            list: List of search results
        """

        web_search_chain_response = self.other_chain.run({"query": absolute_query})

        match = re.search(r"\{.*\}", web_search_chain_response, re.DOTALL)
        if match:
            web_search_chain_response = match.group(0)

        try:
            web_search_chain_response = json.loads(web_search_chain_response)
            web_search_query = web_search_chain_response["response"]
        except Exception as e:
            self.logger.error(f"Error in web search chain: {e}")
            return "An error occurred while processing your query. Please try again."
        
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        search = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list")

        web_search_results = search.invoke(web_search_query)

        return web_search_results


    def get_response(self, query):
        """
        Main method to process user queries and generate appropriate responses.
        
        This method orchestrates the entire conversation flow:
        1. Processes the query with context
        2. Classifies the intent
        3. Extracts relevant entities
        4. Generates follow-up questions
        5. Handles special cases (greetings, web search)
        
        Args:
            query (str): The user's input query
            
        Returns:
            dict: Response containing intent information, entities, and follow-up questions
        """

        absolute_query = self.get_contextual_query_response(query)
        self.logger.info(f"Final query with no contextual references: {absolute_query}")
        
        intent_category, confidence_score = self.get_intent_classification_response(absolute_query)
        self.logger.info(f"Intent category: {intent_category}, Confidence score: {confidence_score}")

        ai_response = {
            "intent_category": intent_category,
            "confidence_score": confidence_score,
        }

        if intent_category == "other":
            web_search_response = self.get_web_search_response(absolute_query)
            self.logger.info(f"Web search completed: {web_search_response}")
            ai_response["web_search_response"] = web_search_response
            return ai_response
        
        elif intent_category == "greetings":
            if "how are you" in absolute_query.lower() or "how you" in absolute_query.lower():
                ai_response["response"] = "I'm good, thank you! How can I help you today?"
            else:
                ai_response["response"] = "Hello! What can I help you with today?"
                
            return ai_response
        
        elif intent_category == "dining":
            intent = DiningIntent()
            keys = intent.get_keys()
        
        elif intent_category == "travel":
            intent = TravelIntent()
            keys = intent.get_keys()
        
        elif intent_category == "gifting":
            intent = GiftingIntent()
            keys = intent.get_keys()
        
        elif intent_category == "cab_booking":
            intent = CabIntent()
            keys = intent.get_keys()

        entities_chain_response = self.get_extracted_entities_response(absolute_query, keys)
        intent.update_info(entities_chain_response)
        self.logger.info(f"Entities updated with extracted values: {entities_chain_response}")

        intent_attributes = intent.get_info()
        for key, value in intent_attributes.items():
            if value is None or value == "None" or value == "" or value == []:
                intent_attributes[key] = "Not Specified"
        ai_response["key_entities"] = intent_attributes

        intent_entities = intent.get_info()

        follow_up_questions = self.get_follow_up_questions(absolute_query, intent_entities)
        self.logger.info(f"Follow up questions: {follow_up_questions}")
        ai_response["follow_up_questions"] = follow_up_questions

        self.logger.info(f"FinalAI response: {ai_response}")

        return ai_response


def main():
    """
    Main function to run the Streamlit chat interface.
    Sets up the chat interface and handles the conversation flow.
    """

    st.title("Swiggy Chat Agent")
    
    # Initialize chat history for streamlit
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                st.json(message["content"])
            else:
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to do?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get bot response
        chat_agent = ChatAgent()
        response = chat_agent.get_response(prompt)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.json(response)

if __name__ == "__main__":
    main()
