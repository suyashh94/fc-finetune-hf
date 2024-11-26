import os
import json
import time
import argparse
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

class CommandGenerator:
    def __init__(self, input_file, output_file, n=2, sleep_interval=1.0, batch_size=10):
        """
        Initializes the CommandGenerator with the given parameters.

        :param input_file: Path to the input JSON file containing function calls.
        :param output_file: Path to the output JSON file to save commands.
        :param n: Number of commands to generate per function call.
        :param sleep_interval: Time to sleep between batches (in seconds).
        :param batch_size: Number of function calls to process before sleeping.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.n = n
        self.sleep_interval = sleep_interval
        self.batch_size = batch_size
        self.data = {}
        self.llm = None

        load_dotenv()
        self.initialize_llm()

    def initialize_llm(self):
        """Initializes the AzureChatOpenAI LLM client."""
        self.llm = AzureChatOpenAI(
            azure_openai_api_base=os.getenv("AZURE_ENDPOINT"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            deployment_name=os.getenv("DEPLOYMENT_NAME"),  # or your deployment name
            openai_api_version=os.getenv("OPENAI_API_VERSION"),  # or your API version
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def load_function_calls(self):
        """Loads function calls from the input JSON file."""
        with open(self.input_file, 'r') as f:
            self.data = json.load(f)

    def generate_command(self, function_call):
        """
        Generates user commands for a given function call.

        :param function_call: The function call string.
        :return: List of generated commands.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    '''You are a car driving command generator that generates human-like commands to execute a given function. You have been given a function execution string like function_name(parameter_name1=parameter_value1, parameter_name2=parameter_value2, ...).\n
You need to write a user command that would cause the function to be called.\n
Assume that the user is asking while driving a car and is using voice commands.
You can be authoritative, commanding, or polite while writing the command.\n
Be as diverse as possible in your commands.\n
'''
                ),
                ("human", "{function_call}"),
            ]
        )

        chain = prompt | self.llm

        all_commands = []

        for _ in range(self.n):
            res = chain.invoke(
                {
                    "function_call": function_call
                }
            )
            all_commands.append(res.content.strip())

        return all_commands

    def generate_commands_for_all_functions(self):
        """Generates commands for all function calls in the data."""
        for function_name in self.data.keys():
            print(f"Generating commands for function: {function_name}")
            self.data[function_name]['user_commands'] = []
            for i, call in enumerate(self.data[function_name]['calls']):
                print(f"Generating commands for call: {call}")
                commands = self.generate_command(call)
                self.data[function_name]['user_commands'].append(commands)

                if (i + 1) % self.batch_size == 0:
                    # Sleep for specified interval after processing a batch
                    time.sleep(self.sleep_interval)

    def save_commands(self):
        """Saves the data with generated commands to the output JSON file."""
        with open(self.output_file, 'w') as f:
            json.dump(self.data, f, indent=4)

    def run(self):
        """Executes the full pipeline: load data, generate commands, and save results."""
        self.load_function_calls()
        self.generate_commands_for_all_functions()
        self.save_commands()

def main():
    parser = argparse.ArgumentParser(description="Generate user commands for function calls using an LLM.")
    parser.add_argument('--input_file', type=str, default='./data/function_calls.json', help='Path to input JSON file containing function calls.')
    parser.add_argument('--output_file', type=str, default='./data/function_calls_with_commands.json', help='Path to output JSON file to save commands.')
    parser.add_argument('--n', type=int, default=2, help='Number of commands to generate per function call.')
    parser.add_argument('--sleep_interval', type=float, default=1.0, help='Time to sleep between batches (in seconds).')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of function calls to process before sleeping.')
    args = parser.parse_args()

    generator = CommandGenerator(
        input_file=args.input_file,
        output_file=args.output_file,
        n=args.n,
        sleep_interval=args.sleep_interval,
        batch_size=args.batch_size
    )
    generator.run()

if __name__ == "__main__":
    main()