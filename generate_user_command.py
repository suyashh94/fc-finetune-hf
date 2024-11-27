import os
import json
import time
import argparse
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

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
        self.load_personas()

    def initialize_llm(self):
        """Initializes the AzureChatOpenAI LLM client."""
        self.llm = AzureChatOpenAI(
            azure_endpoint = os.getenv("AZURE_ENDPOINT"),
                openai_api_key = os.getenv("OPENAI_API_KEY"),
                deployment_name=os.getenv("DEPLOYMENT_NAME"),  # or your deployment
                openai_api_version=os.getenv("OPENAI_API_VERSION"),  # or your api version
                temperature=0.7,
                max_tokens=None,
                timeout=None,
                max_retries=2,
        )

    def load_function_calls(self):
        """Loads function calls from the input JSON file."""
        with open(self.input_file, 'r') as f:
            self.data = json.load(f)

    def load_personas(self):
        personas = ["""
                You are a 28-year-old digital marketing specialist living in a bustling city. You love sleek designs, drive a compact hybrid, and appreciate tech features that make your commute smoother.
            """,
            """
                You are a 67-year-old retired teacher living in a small town. You drive a well-maintained, modest sedan, and you value reliability and comfort.
            """,
            """
                You are a 35-year-old single mother of two, juggling work and kids' activities. You drive a minivan with ample space for car seats, groceries, and sports gear.
            """,
            """
                You are a 30-year-old designer living a flexible lifestyle. You drive a compact car with a stylish, modern look, and love that it’s easy to park in the city.
            """,
            """
                You are a 21-year-old college student studying environmental science. You drive a used electric car and appreciate its low carbon footprint.
            """,
            """
                You are a 42-year-old sales executive in a tech company. You drive a high-end SUV, valuing its power and status, and often entertain clients on the road.
            """,
            """
                You are a 34-year-old landscape designer. You drive a rugged truck, with space for tools and plants, and appreciate durability over aesthetics.
            """,
            """
                You are a 24-year-old part-time driver who works for a food delivery app. You drive a fuel-efficient compact car that’s easy to maneuver and reliable.
            """,
            """
                You are a 70-year-old retiree who enjoys road trips with your spouse. You drive a roomy SUV that’s comfortable for long drives and has space for camping gear.
            """,
            """
                You are a 38-year-old coach, often transporting equipment and students to games. You drive a practical SUV that’s durable and has ample space for gear.
            """,
            """
                You are a 45-year-old finance analyst living in a suburban neighborhood. You drive a standard sedan and value practicality and efficiency.
            """,
            """
                You are a 32-year-old software engineer who loves the latest technology. You drive a new electric car with advanced features and sleek design.
            """,
            """
                You are a 29-year-old outdoor enthusiast who loves road trips and off-road trails. You drive a rugged SUV or Jeep with 4-wheel drive capabilities.
            """,
            """
                You are a 53-year-old veterinarian with your own practice. You drive a reliable crossover with space for supplies, often traveling to see rural clients.
            """,
            """
                You are a 47-year-old real estate agent. You drive a comfortable, stylish sedan or small SUV that’s presentable when meeting clients.
            """,
            """
                You are a 56-year-old professor who commutes to campus. You drive a fuel-efficient car, often loaded with books, and appreciate low maintenance.
            """,
            """
                You are a 40-year-old package delivery worker in the city. You drive a compact van with plenty of storage, designed for easy in-and-out deliveries.
            """,
            """
                You are a 50-year-old construction worker. You drive a sturdy pickup truck, built to haul materials, and value its rugged durability.
            """,
            """
                You are a 38-year-old stay-at-home parent who drives a family-friendly SUV, appreciating the extra space for errands, kids, and gear.
            """,
            """
                You are a 68-year-old retiree who travels around the country with your spouse. You drive a spacious RV, embracing the freedom to explore and stay anywhere.
            """,            
           ]
        
        self.personas = personas
        return 
    
    def generate_command(self, function_call, persona_definition):
        """
        Generates user commands for a given function call.

        :param function_call: The function call string.
        :return: List of generated commands.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                "human","You are a helpful AI assistant, helping a data scientist who is working on a project with a car manufacturer who wants to create an AI powered voice assistant for their cars. "
                ),
                ("human","{persona_definition}"),
                (
                    "human","You have to generate commands that will trigger the given <FUNCTION_CALL>."
                ),
                (
                    "human", "The <FUNCTION_CALL> is: {function_call}"
                ),
                ("human", "You should consider the parameters of the function, understand them and then generate command that will trigger the given function with the given parameter values."),
                ("human","Command should sound human like and should imitate the manner in which humans interact with cars."),
                ("human","You should abstract out parameter names and are not required to use the exact parameter names when stating parameter values for that parameter."),
                ("human","You should also abstract out parameter values and understand the context and meaning, and can use synonyms or similar words that will lead to the same parameter values."),
            ]
        )

        chain = prompt | self.llm

        all_commands = []

        for _ in range(self.n):
            res = chain.invoke(
                {
                    "function_call": function_call,
                    "persona_definition": persona_definition
                }
            )
            # import pdb; pdb.set_trace()
            all_commands.append(res.content.strip())

        return all_commands

    def generate_commands_for_all_functions(self):
        import random
        """Generates commands for all function calls in the data."""
        for function_name in self.data.keys():
            print(f"Generating commands for function: {function_name}")
            self.data[function_name]['user_commands'] = []
            for i, call in enumerate(self.data[function_name]['calls']):
                print(f"Generating commands for call: {call}")
                # select a 3 personas randomly from self.personas
                command_personas = random.sample(self.personas, 3)
                for persona in command_personas:
                    commands = self.generate_command(call, persona)
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
    parser.add_argument('--output_file', type=str, default='./data/function_calls_with_commands_new.json', help='Path to output JSON file to save commands.')
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