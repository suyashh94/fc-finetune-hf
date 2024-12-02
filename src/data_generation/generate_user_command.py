import os
import json
import time
import argparse
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import functions

from user_command_config import correct_command_gen_prompt, CorrectCommandsOutput, \
     missing_values_gen_prompt,\
        CommandType,MissingValuesOutput
load_dotenv()



class CommandGenerator:
    def __init__(self, input_file, output_file, n=2, sleep_interval=1.0, max_threads=4, batch_size=5):
        """
        Initializes the CommandGenerator with the given parameters.

        :param input_file: Path to the input JSON file containing function calls.
        :param output_file: Path to the output JSON file to save commands.
        :param n: Number of commands to generate per function call.
        :param sleep_interval: Time to sleep after every 10 commands (in seconds).
        :param max_threads: Maximum number of threads for parallel processing.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.n = n
        self.sleep_interval = sleep_interval
        self.max_threads = max_threads
        self.data = {}
        self.llm = None
        self.batch_size = batch_size

        self.global_request_count = 0
        self.lock = threading.Lock()

        self.initialize_llm()
        self.load_personas()

    def initialize_llm(self):
        """Initializes the AzureChatOpenAI LLM client."""
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            deployment_name=os.getenv("DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            temperature=0.9,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def load_function_calls(self):
        """Loads function calls from the input JSON file."""
        with open(self.input_file, 'r') as f:
            self.data = json.load(f)

    def load_personas(self):
        """Loads predefined personas."""
        self.personas = []

    def check_rate_limit(self):
        """Checks and enforces rate limiting globally."""
        with self.lock:
            self.global_request_count += 1
            if self.global_request_count % self.batch_size == 0:
                print(f"Generated {self.global_request_count} commands. Pausing for {self.sleep_interval} seconds...")
                time.sleep(self.sleep_interval)

    def generate_command(self, function_call, type=CommandType.CORRECT_COMMANDS):
        """Generates user commands for a given function call."""
        
        fn_name = function_call.split('(')[0]
        fn = list(filter(lambda x: x['name'] == fn_name, functions))[0]
        params = list(fn['parameters']['properties'].keys())
        description = fn['description']
        
        if type == CommandType.CORRECT_COMMANDS:
            prompt_message = correct_command_gen_prompt
            parser = CorrectCommandsOutput
        elif type == CommandType.MISSING_VALUES_COMMANDS:
            prompt_message = missing_values_gen_prompt
            parser = MissingValuesOutput
        
        prompt = ChatPromptTemplate.from_messages(
             prompt_message
        )

        chain = prompt | self.llm.with_structured_output(parser)
        all_commands = []
        
        self.check_rate_limit()  # Enforce global rate limit per command
        res = chain.invoke(
            {
                "function_call": function_call,
                # "function_params": ', '.join(params),
                # "function_description": description
             }
        )
        if type == CommandType.CORRECT_COMMANDS:
            all_commands = res.commands
        elif type == CommandType.MISSING_VALUES_COMMANDS:
            all_commands = res.result
            all_commands = [c.dict() for c in all_commands]
        
        return all_commands

    def generate_commands_for_function(self, function_name, type=CommandType.CORRECT_COMMANDS):
        """Generates commands for all calls of a single function."""
        function_data = self.data[function_name]
        function_data["user_commands_{}".format(type.name)] = []  # Initialize the user_commands key

        for call in function_data["calls"]:
            commands = self.generate_command(call, type=type)
            function_data["user_commands_{}".format(type.name)].append(commands)
            # break

    def generate_commands_for_all_functions(self, parallel=True, type=CommandType.CORRECT_COMMANDS):
        # import pdb; pdb.set_trace()
        """Generates commands for all functions using multithreading."""
        if parallel:
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                futures = [
                    executor.submit(self.generate_commands_for_function, function_name, type)
                    for function_name in self.data.keys()
                ]
                for future in futures:
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"Error in generating commands: {exc}")
        else:
            for function_name in self.data.keys():
                self.generate_commands_for_function(function_name, type=type)

    def save_commands(self):
        """Saves the data with generated commands to the output JSON file."""
        # import pdb; pdb.set_trace()
        with open(self.output_file, 'w') as f:
            json.dump(self.data, f, indent=4)

    def run(self, parallel=True, type=CommandType.CORRECT_COMMANDS):
        """Executes the entire command generation pipeline."""
        self.load_function_calls()
        self.generate_commands_for_all_functions(parallel=parallel, type=type)
        self.save_commands()

def main():
    parser = argparse.ArgumentParser(description="Generate user commands for function calls using an LLM.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output JSON file.')
    parser.add_argument('--n', type=int, default=1, help='Number of commands to generate per function call.')
    parser.add_argument('--sleep_interval', type=float, default=1.0, help='Time to sleep after every 10 commands.')
    parser.add_argument('--max_threads', type=int, default=4, help='Maximum number of threads for parallel processing.')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of commands to generate before pausing.')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing.')
    parser.add_argument('--type', type=str, default='correct_commands', help='Type of commands to generate.', \
        choices=['correct_commands', 'missing_values_commands'])
    
    
    args = parser.parse_args()
    
    if args.type == 'correct_commands':
        type = CommandType.CORRECT_COMMANDS
    elif args.type == 'missing_values_commands':
        type = CommandType.MISSING_VALUES_COMMANDS

    generator = CommandGenerator(
        input_file=args.input_file,
        output_file=args.output_file,
        n=args.n,
        sleep_interval=args.sleep_interval,
        max_threads=args.max_threads,
        batch_size=args.batch_size,
    )
    generator.run(args.parallel, type=type)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution completed in {time.time() - start_time:.2f} seconds.")