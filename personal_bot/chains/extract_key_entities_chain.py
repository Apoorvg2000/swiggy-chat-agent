from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import sys
sys.path.append("./personal_bot")
from ..get_llm import get_llm
from ..get_memory import BotMemory

def extract_key_entities_chain():
    llm = get_llm("llama-3.3-70b-versatile", temperature=0.5)
    memory = BotMemory().get_memory()
    
    extract_key_entities_prompt = PromptTemplate(
        input_variables=["input"],
        template="""Instructions:
You are an intelligent entity extraction assistant. Your job is to extract key entities from a user's natural language input, given a list of key entities and the user's natural language input.

You will be given two inputs:

1. A list of key entities (e.g., ['date', 'time', 'location', 'budget', 'cuisine', 'party_size', 'special_requests'])  
2. A user input sentence.

Your task is to extract all entities from the user input that are present in the given list of key entities.

Rules:

- Extract only entities present in the given list.  
- If an entity is not mentioned or unclear, omit it.  
- The output must be a JSON object with keys only for entities found.  
- The key entity `special_requests`, must always be a list of strings if present; otherwise omit it.  
- For numeric entities like `party_size` or `members`, convert all numeric values written as words (e.g., "two", "five") into numbers (e.g., 2, 5).  
- Dates and times can be extracted as natural language strings (e.g., "tomorrow evening", "9 PM").  
- Locations, cuisine, recipient and occasion should always be strings.
- Do not assume any information if not explicitly provided.
- Only extract entities that are present in the Key Entities list provided.
- Do not add any explanation or extra text, output only valid JSON.

Examples:

Example 1:

Key Entities: ['date', 'time', 'location', 'budget', 'cuisine', 'party_size', 'special_requests']
User: "Need a sunset-view table for two tonight; gluten-free menu a must"

Response:
{{
  "party_size": "2",
  "date": "tonight",
  "special_requests": ["sunset-view table", "gluten-free menu"]
}}

Example 2:

Key Entities: ['location_from', 'location_to', 'start_date', 'end_date', 'mode', 'members', 'budget', 'special_requests']
User: "Planning a trip from Delhi to Goa for five members from 10th June to 15th June, budget 50000 INR"

Response:
{{
  "location_from": "Delhi",
  "location_to": "Goa",
  "members": "5",
  "start_date": "10th June",
  "end_date": "15th June",
  "budget": "50000 INR"
}}

Example 3:

Key Entities: ['date', 'time', 'location', 'budget', 'cuisine', 'party_size', 'special_requests']
User: "Book a table for four people at Olive Garden on Friday evening"

Response:
{{
  "party_size": "4",
  "location": "Olive Garden",
  "date": "Friday evening"
}}

Example 4:

Key Entities: ['pickup_location', 'drop_off_location', 'members', 'budget', 'special_requests']
User: "Book a cab from airport to hotel for three people, budget 500 INR, need a baby seat"

Response:
{{
  "pickup_location": "airport",
  "drop_off_location": "hotel",
  "members": "3",
  "budget": "500 INR",
  "special_requests": ["baby seat"]
}}

Example 5:

Key Entities: ['recipient', 'occasion', 'budget', 'special_requests']
User: "Gift for mom on Mother's Day, budget 2000 INR, something handmade preferred"

Response:
{{
  "recipient": "mom",
  "occasion": "Mother's Day",
  "budget": "2000 INR",
  "special_requests": ["something handmade"]
}}

Example 6:

Key Entities: ['location_from', 'location_to', 'start_date', 'end_date', 'mode', 'members', 'budget', 'special_requests']
User: "Looking for a flight on 25th May, traveling alone"

Response:
{{
  "start_date": "25th May",
  "members": "1",
  "mode": "flight"
}}

Example 7:

Key Entities: ['recipient', 'occasion', 'budget', 'special_requests']
User: "Looking for birthday gift under 1000, no specific recipient"

Response:
{{
  "occasion": "birthday",
  "budget": "1000"
}}

Example 8:

Key Entities: ['pickup_location', 'drop_off_location', 'members', 'budget', 'special_requests']
User: "Need a ride from office to home at 8 pm tonight"

Response:
{{
  "pickup_location": "office",
  "drop_off_location": "home",
  "members": "1",
  "special_requests": ["8 pm tonight"]
}}

Strictly follow the above rules and examples to ensure 100% classification accuracy.

{input}

"""
    )

    extract_key_entities_chain = LLMChain(llm=llm, prompt=extract_key_entities_prompt, memory=memory)
    
    return extract_key_entities_chain

if __name__ == "__main__":
    response = extract_key_entities_chain().invoke("I want to book a cab from Mumbai to Delhi")
    print(response)
