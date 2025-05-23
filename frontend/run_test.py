import json
import os
import argparse
from chat_agent import ChatAgent

def run_tests(test_cases_path, output_dir):
    # Initialize the chat agent
    chat_agent = ChatAgent()
    
    # Read test cases
    with open(test_cases_path, 'r') as f:
        test_data = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output file path
    output_file = os.path.join(output_dir, 'test_results.json')
    
    # Process each test case
    results = []
    for test_case in test_data['test_cases']:
        test_id = test_case['id']
        user_input = test_case['input']
        expected_intent = test_case['intent']
        
        # Get response from chat agent
        try:
            response = chat_agent.get_response(user_input)
        except Exception as e:
            print(f"Error in test case {test_id}: {e}")
            response = str(e)
            break
        
        # Store the result
        result = {
            'id': test_id,
            'input': user_input,
            'output': response
        }
        results.append(result)
        with open(output_file, 'w') as f:
            json.dump({'test_results': results}, f, indent=4)
    
    print(f"Test results have been saved to {output_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run chat agent tests with specified test cases file and output directory')
    parser.add_argument('--test-cases', '-t', 
                      default='../tests/test_cases.json',
                      help='Path to the test cases JSON file (default: ../tests/test_cases.json)')
    parser.add_argument('--output-dir', '-o',
                      default='../test_results',
                      help='Directory to save test results (default: ../test_results)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run tests with provided arguments
    run_tests(args.test_cases, args.output_dir)

if __name__ == "__main__":
    main()
