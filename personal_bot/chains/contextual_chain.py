from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import sys
sys.path.append("./personal_bot")
from ..get_llm import get_llm
from ..get_memory import BotMemory

def contextual_query_chain():
    llm = get_llm("llama-3.3-70b-versatile", temperature=0.3)
    memory = BotMemory().get_memory()

    centextual_query_prompt = PromptTemplate(
            input_variables=["input"],
            template="""Instructions:
You are a highly intelligent chatbot that recognizes and replaces contextual words in queries with fully self-contained terms using the last five messages (combined from both user and assistant). Your goal is to ensure that all references are explicit and unambiguous before processing.

Guidelines:

1. Detect Contextual References: Identify words or phrases like "that feature," "those colors," "it," or "such options" that depend on previous messages for clarity.
2. Retrieve Relevant Context: Extract the most relevant details from the last five messages (both user and assistant) that clarify the ambiguous references.
3. Replace Contextual Words: Substitute only the ambiguous references with explicit details from the retrieved context while keeping the rest of the query unchanged.

Ensure Clarity: The final query should be fully understandable on its own without requiring any external context.

Handle Edge Cases:

1. IMPORTANT: If the query has no contextual references or words, return it unchanged.
2. If the query contains contextual references or words but no prior context or messages are provided, return it as is.
3. If there is no relevant context, respond with "Unable to determine context. Please provide more details."
4. Also if the user makes some spelling or grammatical errors, correct them in your response.
5. Only and strictly return the transformed query and NOT the response of the query.

Maintain Original Meaning: Ensure the transformed query preserves the intent of the user’s original question and only replace the contextual words without adding any extra words.

Examples:

Example 1: Contextual Reference to an Earlier User Query

a.

Context:
User: "Suggest some romantic rooftop restaurants."

query: "Can you book one for 7 PM today?"

Response:
{{
    "response": "Can you book a romantic rooftop restaurant for 7 PM today?"
}}

b.

Context:
User: "I want to dine with my parents near MG Road."

query: "My budget is 2000 rupees."

Response:
{{
    "response": "My budget is 2000 rupees for dining with my parents near MG Road."
}}

c.

Context:
User: "Suggest a few weekend getaways near Mumbai."

query: "Can you plan one of these for this weekend?"

Response:
{{
    "response": "Can you plan a weekend getaway near Mumbai for this weekend?"
}}

d.

Context:
User: "I want to gift something meaningful for under 2000."

query: "Add the mug and diary, forget the portrait."

Response:
{{
    "response": "Add the mug and diary to the gift options under 2000 rupees, skip the portrait."
}}

e.

Context:
User: "Book a cab to the railway station at 9 AM."

query: "Make it an SUV instead."

Response:
{{
    "response": "Book an SUV cab to the railway station at 9 AM."
}}

Example 2: Query Without Clear Context

a.

Context:
User: "How do I update my mobile number on Aadhaar?"

query: "Can you book cab to my destination for me?"

Response:
{{
    "response": "Unable to determine context. Please provide more details."
}}

b.

Context:
User: "Show me the driving license renewal process."

query: "Is that better than going in person?"

Response:
{{
    "response": "Unable to determine context. Please provide more details."
}}

Example 3: Contextual Query Without Any Context, return the query as it is

Context:

query: "What documents do I need?"

Response:
{{
    "response": "What documents do I need?"
}}

Final Output Format:
Your output should only be the the query with no contextual words (if applicable), otherwise, return a natural response. Return it strictly as a JSON object with the key "response" as shown in the examples above. There should be no extra words before or after the JSON object.

{input}

"""
    )

    contextual_chain = LLMChain(llm=llm, prompt=centextual_query_prompt, memory=memory)
    
    return contextual_chain