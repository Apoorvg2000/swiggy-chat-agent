from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import sys
sys.path.append("./personal_bot")
from ..get_llm import get_llm
from ..get_memory import BotMemory

def intent_classifier_chain():
    llm = get_llm(model_name="llama-3.3-70b-versatile", temperature=0.3)
    memory = BotMemory().get_memory()


    intent_classification_prompt = PromptTemplate(
        input_variables=["query"],
        template="""Instructions:
You are an intelligent AI assistant. Your task is to classify a user's natural language input into one of the following categories:

- "dining" - strictly for queries related to making reservations at a restaurant or a dining outlet.
- "travel" - strictly for queries related to flights, train, or planning a trip, etc.
- "gifting" - strictly for queries related to gifting someone.
- "cab_booking" - strictly for queries related to booking a cab.
- "other" (for anything that doesn't clearly fit the above) - for queries that require searching the web, or asking for help with a task, etc.
- "greetings" - strictly for queries where user provides only greetings.

Also, estimate your confidence level as a float between 0 (very uncertain) and 1 (very confident) based on how clear and relevant the query is to the intent. The confidence score should be a measure of your certainty in the classification.

NOTE: Output your answer strictly in this JSON format:

{{
  "intent_category": "<one of: dining, travel, gifting, cab_booking, other>",
  "confidence_score": <float between 0 and 1>
}}

DO NOT include any explanation or text outside the JSON.

Examples:

Example 1:
User: "Hello there, need a table for two by the beach around 7 PM tonight, vegetarian menu if possible"

Response:
{{
  "intent_category": "dining",
  "confidence_score": 0.93
}}

Example 2:
User: "Planning a trip from Delhi to Manali for the long weekend, 4 people, budget-friendly options please"

Response:
{{
  "intent_category": "travel",
  "confidence_score": 0.91
}}

Example 3:
User: "Hey, I want to send a gift to my sister for her graduation – something thoughtful under 1000 rupees"

Response:
{{
  "intent_category": "gifting",
  "confidence_score": 0.89
}}

Example 4:
User: "I need a cab from airport to hotel around 10:30 AM"

Response:
{{
  "intent_category": "cab_booking",
  "confidence_score": 0.95
}}

Example 5:
User: "How do I update the address on my Aadhaar card?"

Response:
{{
  "intent_category": "other",
  "confidence_score": 0.97
}}

Example 6:
User: "Can you find a quiet place for dinner near my office tonight?"

Response:
{{
  "intent_category": "dining",
  "confidence_score": 0.88
}}

Example 7:
User: "Hi, I want to escape the city this weekend, maybe somewhere in the mountains"

Response:
{{
  "intent_category": "travel",
  "confidence_score": 0.84
}}

Example 8:
User: "I want to surprise my dad with something meaningful on his birthday"

Response:
{{
  "intent_category": "gifting",
  "confidence_score": 0.86
}}

Example 9:
User: "Need to get from the office to the train station by 6"

Response:
{{
  "intent_category": "cab_booking",
  "confidence_score": 0.81
}}

Example 10:
User: "find some cool places to hangout"

Response:
{{
  "intent_category": "other",
  "confidence_score": 0.68
}}

Example 11:
User: "Hey, what are the rules for carrying liquids on domestic flights?"

Response:
{{
  "intent_category": "other",
  "confidence_score": 0.89
}}

Example 12:
User: "Hi, how are you?"

Response:
{{
  "intent_category": "greetings",
  "confidence_score": 0.98
}}

Example 13:
User: "what are some good travel destinations i could explore in summer"

Response:
{{
  "intent_category": "other",
  "confidence_score": 0.86
}}

Now, classify the following user input:

User: {query}

"""
    )

    
    intent_classification_chain = LLMChain(llm=llm, prompt=intent_classification_prompt, memory=memory)

    return intent_classification_chain