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

from user_command_config import complete_command_gen_prompt, CorrectCommandsOutput, \
     incomplete_command_gen_prompt,\
        CommandType,MissingValuesOutput,IncompleteCommands,\
            incomplete_command_gen_prompt_reinforced, IncompleteCommandOutput,\
                SampleCorrectnessJudgement, incorrectness_judgement_prompt
            
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

    def generate_command(self, function_call):
        """Generates user commands for a given function call."""
        
        fn_name = function_call.split('(')[0]
        fn = list(filter(lambda x: x['name'] == fn_name, functions))[0]
        params = list(fn['parameters']['properties'].keys())
        description = fn['description']
        
        
        prompt_message = complete_command_gen_prompt
        parser = CorrectCommandsOutput
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
        all_commands = res.commands
        all_incomplete_commands = []
        
        incomplete_prompt_message = incomplete_command_gen_prompt_reinforced
        incomplete_parser = IncompleteCommandOutput
        incomplete_prompt = ChatPromptTemplate.from_messages(
                incomplete_prompt_message
            )
        
        incomplete_chain = incomplete_prompt | self.llm.with_structured_output(incomplete_parser)
        
        for command in all_commands:
            self.check_rate_limit()  # Enforce global rate limit per command
            try:
                res = incomplete_chain.invoke(
                    {
                        "function_call": function_call,
                        "command": command
                    }
                )
                incomplete_command = res.incomplete_command
                modified_incorrect_function_call = res.modified_incorrect_function_call
                print(f"Complete command: {command} for function call: {function_call}")
                print(f"Incomplete command: {incomplete_command} with modified incorrect function call: {modified_incorrect_function_call}")
                all_incomplete_commands.append({"incomplete_command": incomplete_command, "modified_incorrect_function_call": modified_incorrect_function_call})
            except Exception as e:
                print(f"Error in generating incomplete command: {e}")
                all_incomplete_commands.append({"incomplete_command": "", "modified_incorrect_function_call": ""})
            
        return all_commands, all_incomplete_commands

    def generate_commands_for_function(self, function_name):
        """Generates commands for all calls of a single function."""
        function_data = self.data[function_name]
        function_data["complete_commands"] = []  # Initialize the user_commands key
        function_data["incomplete_commands"] = []  # Initialize the user_commands key

        for i,call in enumerate(function_data["calls"]):
            complete_commands, incomplete_commands = self.generate_command(call)
            function_data["complete_commands"].append(complete_commands)
            function_data["incomplete_commands"].append(incomplete_commands)
            
            # if i == 1:
            #     break

    def generate_commands_for_all_functions(self, parallel=True):
        # import pdb; pdb.set_trace()
        """Generates commands for all functions using multithreading."""
        if parallel:
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                futures = [
                    executor.submit(self.generate_commands_for_function, function_name)
                    for function_name in self.data.keys()
                ]
                for future in futures:
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"Error in generating commands: {exc}")
        else:
            for function_name in self.data.keys():
                self.generate_commands_for_function(function_name)

    def is_negative_sample_correct(self, incomplete_user_command, modified_incorrect_function_call, parameters):
        
        prompt_message = incorrectness_judgement_prompt
        parser = SampleCorrectnessJudgement
        prompt = ChatPromptTemplate.from_messages(
             prompt_message
        )
        self.llm.temperature = 0
        chain = prompt | self.llm.with_structured_output(parser)
        res = chain.invoke(
            {
                "incomplete_user_command": incomplete_user_command,
                "modified_incorrect_function_call": modified_incorrect_function_call,
                "parameters": parameters
             }
        )
        return res.judgement, res.reason
        
        # return True
    
    def validate_negative_samples_for_function(self, function_name):
        """Validates negative samples for a single function."""
        function_data = self.data[function_name]
        incomplete_calls = function_data["incomplete_commands"]
        complete_calls = function_data["complete_commands"]
        fn_calls = function_data["calls"]
        
        
        fn = list(filter(lambda x: x['name'] == function_name, functions))[0]
        params = list(fn['parameters']['properties'].keys())
        
        for i, call in enumerate(incomplete_calls):
            fn_call = fn_calls[i]
            complete_commads = complete_calls[i]
            for j, incomplete_call in enumerate(call):
                inc_command = incomplete_call["incomplete_command"]
                inc_fn_call = incomplete_call["modified_incorrect_function_call"]
                com_fn_call = fn_calls[i]
                com_command = complete_commads[j]
                is_correct, reason = self.is_negative_sample_correct(inc_command, inc_fn_call, params)
                incomplete_calls[i][j]["is_correct"] = is_correct
                incomplete_calls[i][j]["reason"] = reason
        
        function_data["incomplete_commands"] = incomplete_calls    
    
    def validate_negative_samples_for_all_functions(self, parallel=True):
        """Validates negative samples for all functions using multithreading."""
        if parallel:
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                futures = [
                    executor.submit(self.validate_negative_samples_for_function, function_name)
                    for function_name in self.data.keys()
                ]
                for future in futures:
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"Error in validating negative samples: {exc}")
        else:
            for function_name in self.data.keys():
                self.validate_negative_samples_for_function(function_name)
    
    def save_commands(self):
        """Saves the data with generated commands to the output JSON file."""
        # import pdb; pdb.set_trace()
        with open(self.output_file, 'w') as f:
            json.dump(self.data, f, indent=4)

    def run(self, parallel=True, type=CommandType.CORRECT_COMMANDS):
        """Executes the entire command generation pipeline."""
        self.load_function_calls()
        
        self.generate_commands_for_all_functions(parallel=parallel)
        
        # self.validate_negative_samples_for_all_functions(parallel=parallel)
        
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
    
    
    
    args = parser.parse_args()

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