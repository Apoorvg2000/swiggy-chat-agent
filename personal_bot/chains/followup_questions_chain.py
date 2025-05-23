from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import sys
sys.path.append("./personal_bot")
from ..get_llm import get_llm
from ..get_memory import BotMemory

def followup_questions_chain():
    llm = get_llm("llama-3.3-70b-versatile", temperature=0.7, max_tokens=2500)
    memory = BotMemory().get_memory()

    followup_questions_prompt = PromptTemplate(
        input_variables=["input"],
        template="""Instructions:
You are a smart assistant that helps collect missing or unclear details from users based on their requests. Some key information has been extracted into a dictionary from the user's request. Your job is to generate clear and concise follow-up questions to gather any missing or ambiguous information from the user.

Instructions:

1. A user_query – the original message from the user.
2. An info dictionary – contains extracted key fields with values that might be:
   - Filled with clear data,
   - None (missing),
   - Ambiguous or unclear data (e.g., "something spicy", "soon", "a few people", "my office", etc.)

Your task is:

1. Review each field in the dictionary.
2. For fields with value = None, generate a follow-up question to ask for that information.
3. For fields with vague, unclear or ambiguous values, ask the user to clarify.
4. If a field is already filled and clearly understood, skip it — DO NOT ask again.
5. If the user provides the location, be it destination, pickup or drop off location, in an ambiguos manner (eg: my place, friend's house, office, etc.), ask for the exact location.
6. Ask questions in a natural, friendly tone, based on the user’s original query.
7. Do not add any explanation or extra text, output only valid JSON.

Output your response strictly in this JSON format:

{{
  "response": ["question1", "question2", "question3"]
}}

Examples:

Example 1:
User: "I'd like to book a nice dinner for tomorrow; party of 4; a nice lake view is preferred."

Info:
{{
    "date": "tomorrow",
    "time": None,
    "location": None,
    "budget": None,
    "cuisine": None,
    "party_size": None,
    "special_requests": ["a nice lake view"]
}}

Response:
{{
  "response": [
        "What time would you like to book the dinner for?",
        "Where would you prefer to dine? Would you prefer a particular restaurant?",
        "How many people will be attending?",
        "Do you have a preferred cuisine or type of food in mind?",
        "Do you have a specific budget in mind for the dinner?",
    ]
}}

Example 2:
User: "Plan a trip to Goa with friends this month under 50000 INR."

Info:
{{
    "location_from": None,
    "location_to": "Goa",
    "start_date": "this month",
    "end_date": None,
    "mode": None,
    "members": "friends",
    "budget": "under 50000 INR",
    "special_requests": None
}}

Response:
{{
  "response": [
        "Where will you be travelling from?",
        "Can you specify the exact start and end dates for the trip?",
        "How many friends will be travelling?",
        "Do you have a preferred mode of travel (flight, train, etc.)?",
        "Do you have any special requests or preferences for the trip?"
    ]
}}


Example 3:
User: "Reserve a table for five at an Indian restaurant near Bandra tonight."

Info:
{{
    "date": "today",
    "time": "tonight",
    "location": "Bandra",
    "budget": None,
    "cuisine": "Indian",
    "party_size": 5,
    "special_requests": None
}}

Response:
{{
  "response": [
        "Do you have a specific budget in mind for the dinner?",
        "Any special requests for the reservation?"
    ]
}}

Example 4:
User: "Get me a cab to the airport for 500 Rs. make sure baby seat is there."

Info:
{{
    "pickup_location": None,
    "drop_off_location": "airport",
    "members": None,
    "budget": "500 Rs.",
    "special_requests": ["baby seat is there"]
}}

Response:
{{
    "response": [
        "Where should the cab pick you up from?",
        "Could you please provide more details about your drop off location? Which airport are you travelling to?",
        "How many people will be travelling?",
    ]
}}

Example 5:
User: "I want to go from Mumbai to Delhi by train next Monday."

Info:
{{
    "location_from": "Mumbai",
    "location_to": "Delhi",
    "start_date": "next Monday",
    "end_date": None,
    "mode": "train",
    "members": "1",
    "budget": None,
    "special_requests": None
}}

Response:
{{
  "response": [
        "Can you specify the return date for the trip?",
        "Do you have a budget in mind for the journey?",
        "Any specific requests or preferences during the travel?"
    ]
}}

Example 6:
User: "I need something for my sister's graduation."

Info:
{{
    "recipient": "sister",
    "occasion": "graduation",
    "budget": "not too expensive",
    "special_requests": None
}}

Response:
{{
    "response": [
        "Could you specify a price range you consider 'not too expensive'?",
        "Any special requests or preferences for the gift?"
    ]
}}

Example 8:
User: "Book a car to the airport from my place; 3 people."

Info:
{{
    "pickup_location": "my place",
    "drop_off_location": "airport",
    "members": "3",
    "budget": None,
    "special_requests": None
}}

Response:
{{
    "response": [
        "Could you please specify your exact pick you up location?",
        "Could you please provide more details about your drop off location? Which airport are you travelling to?",
        "What is the budget for the car?",
        "Do you have any preferences or special requests for the car?"
    ]
}}

Stricly follow above instructions and examples and output only valid JSON.

{input}

"""
    )

    followup_questions_chain = LLMChain(
        llm=llm,
        prompt=followup_questions_prompt,
        memory=memory
    )

    return followup_questions_chain