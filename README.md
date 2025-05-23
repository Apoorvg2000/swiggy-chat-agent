# Chat Agent System

A sophisticated chat agent system capable of handling multiple intents including dining reservations, travel planning, cab bookings, and gift sending. The system uses advanced natural language processing to understand user queries and provide relevant responses. In this chat agent, I have used Groq with LangChain due to its ease of use and smooth integration with LangChain.

## Features

- **Multi-Intent Support**: Handles various user intents including:
  - Dining reservations
  - Travel planning
  - Cab bookings
  - Gift sending
  - General queries

- **Context Awareness**: Maintains conversation context for better user interaction
- **Entity Extraction**: Extracts relevant information from user queries
- **Follow-up Questions**: Generates contextual follow-up questions
- **Web Search Integration**: Handles general queries through web search
- **Error Handling**: Robust error handling and user-friendly error messages

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Apoorvg2000/swiggy-chat-agent.git
cd swiggy-chat-agent
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python3 -m venv myenv

# Activate virtual environment

# On Windows:
myenv\Scripts\activate

# On macOS/Linux:
source myenv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Add Groq API key in evironment file .env
```bash
GROQ_API_KEY=<your_groq_api_key>
```

## Running the Chat Agent

### Using Streamlit Interface

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Run the Streamlit app:
```bash
streamlit run chat_agent.py
```

3. Open your browser and go to `http://localhost:8501`

### Using the Chat Agent Programmatically

```python
from chat_agent import ChatAgent

# Initialize the chat agent
chat_agent = ChatAgent()

# Get response for a user query
response = chat_agent.get_response("Book a table for 4 people at an Italian restaurant")
print(response)
```

## Running Tests

The repository includes a test runner (`run_test.py`) that can execute test cases and generate detailed results.

### Basic Usage

1. Navigate to frontend directory
```bash
cd frontend
```

2. Now you can run the tests on your custom test cases. Provide the test cases file path and the output directory path as shown below

```bash
python run_test.py --test-cases path/to/test_cases.json --output-dir path/to/output
```

Options:
- `--test-cases` or `-t`: Path to test cases JSON file
- `--output-dir` or `-o`: Directory to save test results

## Test Cases

The test suite includes 51 test cases covering various scenarios:

### Intent Distribution
- Dining Intent: 20 test cases
- Travel Intent: 12 test cases
- Cab Booking Intent: 10 test cases
- Gifting Intent: 7 test cases
- Other/General Queries: 2 test cases

### Test Categories
1. **Basic Functionality Tests**
   - Simple queries with complete information
   - Single intent queries
   - Clear entity specifications

2. **Edge Cases**
   - Invalid dates/times
   - Unrealistic budgets
   - Impossible locations
   - Very large groups
   - Very short notice requests

3. **Complex Scenarios**
   - Multiple intents in one query
   - Complex travel itineraries
   - Corporate bookings
   - International deliveries
   - Special requirements

4. **Error Handling**
   - Invalid inputs
   - Missing information
   - Conflicting preferences
   - Unrealistic requests

## Test Results Format

The test results are saved in JSON format with the following structure:

```json
{
    "test_results": [
        {
            "id": "TC001",
            "input": "User query",
            "expected_intent": "intent_type",
            "output": {
                // Chat agent response
            }
        }
    ]
}
```

## Test Results

### Intent Classification
- Total Test Cases: 51
- Correctly Classified: 49
- Accuracy: 96.08%
- Incorrect Classifications: 2 (due to context removal chain)

### Entity Extraction
- Success Rate: 100%
- All entities were correctly extracted from input queries
- No missing or incorrect entity extractions

### Response Quality
1. **Follow-up Questions**
   - Issue: Some queries generated unnecessary follow-up questions
   - Impact: Reduced user experience efficiency
   - Example: Asking for additional information when sufficient details were already provided. For example: the agent asked for more preferences or special requests, when user already provided 1-2 special requests.

2. **Context Handling**
   - Issue: 2 queries were incorrectly rewritten due to context removal chain
   - Impact: Minor degradation in response accuracy
   - Areas for Improvement: Context preservation mechanism

### Performance Summary
- High accuracy in intent classification (96.08%)
- Perfect entity extraction (100%)
- Room for improvement in:
  - Follow-up question generation
  - Context preservation
  - Response optimization

## Performance Metrics

The chat agent's performance is evaluated based on:
1. Intent Classification Accuracy
2. Entity Extraction Accuracy
3. Response Relevance
4. Error Handling Effectiveness
