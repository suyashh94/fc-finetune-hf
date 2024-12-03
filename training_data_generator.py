import os
import random
import re
import json
import argparse
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
from config import functions

class TrainingDataGenerator:
    def __init__(self, prompt_insertion=False, output_file='car_finetuning_gpt', read_data_from_file=False, data_file=None):
        """
        Initializes the TrainingDataGenerator with the given parameters.

        :param prompt_insertion: Whether to insert prompts for each function.
        :param output_file: Base name for the output files.
        :param read_data_from_file: Whether to read data from a file.
        :param data_file: Path to the data file if read_data_from_file is True.
        """
        self.prompt_insertion = prompt_insertion
        self.output_file = output_file
        self.read_data_from_file = read_data_from_file
        self.data_file = data_file
        self.functions = functions
        self.messages = []
        self.questions_dict = self.initialize_questions_dict()
        self.fake = Faker()

    @staticmethod
    def initialize_questions_dict():
        """
        Initializes the dictionary containing questions for each function.

        :return: A dictionary mapping function names to lists of questions.
        """
        return {
            "adjust_temperature": [
                "Can you set the {zone} temperature to {temperature} degrees?",
                "Please adjust the {zone} zone temperature to {temperature}.",
                "I need the {zone} area to be {temperature} degrees.",
                "Set the {zone} temperature to {temperature} degrees Celsius.",
                "Adjust the {zone} zone to {temperature} degrees, please.",
                "Adjust temperature to {temperature}",
                "Set temperature to {temperature}"
            ],
            # ... Add other functions and their questions here
            # (Include all entries from the original `questions_dict`)
            "adjust_seat": [
                "Can you move the seat {position}?",
                "Please adjust the seat to {position}.",
                "I need the seat adjusted {position}.",
                "Push the seat {position}.",
                "Please adjust the {seat_type}'s seat to {position}.",
                "I need the {seat_type}'s seat adjusted {position}.",
                "Set the {seat_type}'s seat {position}.",
                "Adjust the {seat_type} seat {position} and remember this setting."
            ],
            # Include all other entries...
        }

    def get_random_question(self, function_name):
        """
        Gets a random question for a given function name.

        :param function_name: Name of the function.
        :return: A random question string.
        """
        question_list = self.questions_dict.get(function_name)
        if question_list:
            return random.choice(question_list)
        else:
            return ""

    def get_random_value(self,param, data_type):
        fake = self.fake

        if data_type == "boolean":
            return random.choice([True, False])
        elif param == "date":
            return (datetime.now() + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
        elif param == "time":
            hour = random.randint(1, 12)
            minute = random.randint(0, 59)
            period = random.choice(["AM", "PM"])
            return f"{hour}:{minute:02} {period}"
        elif param in ["weight", "original_price", "loan_amount"]:
            if param == "weight":
                return random.randint(60, 120)
            elif param == "original_price" or param == "loan_amount":
                return random.randint(500, 10000)
        elif param in ["height", "password_length", "term_years", "angle"]:
            if param == "height":
                return random.randint(150, 200)
            elif param == "password_length" or param == "term_years":
                return random.randint(5, 30)
            elif param == "angle":
                return random.randint(0, 180)
        elif param in ["age", "credit_score", "user_rating"]:
            if param == "age":
                return random.randint(18, 80)
            elif param == "credit_score":
                return random.randint(300, 850)
            elif param == "user_rating":
                return random.randint(1, 10)
        elif param == "gender":
            return random.choice(["Male", "Female", "Other"])
        elif param == "servings":
            return random.randint(1, 12)
        elif param in ["income", "monthly_contribution", "current_savings"]:
            if param == "income":
                return random.randint(30000, 200000)
            elif param == "monthly_contribution":
                return random.uniform(100, 2000)
            elif param == "current_savings":
                return random.uniform(5000, 500000)
        elif param in ["interest_rate", "discount_percentage"]:
            return round(random.uniform(1.0, 15.0), 2)
        elif param in ["customer_id", "movie_id"]:
            return random.randint(1000, 9999)
        elif param == "participants":
            return [fake.name() for _ in range(random.randint(2, 10))]
        elif param == "title":
            return fake.sentence(nb_words=4).replace(".", "")
        elif param in ["recipient_name", "contact_name"]:
            return fake.name()
        elif param == "subject":
            return fake.sentence(nb_words=4)
        elif param == "body":
            return fake.paragraph(nb_sentences=3)
        elif param == "query":
            return fake.sentence(nb_words=5)
        elif param == "restaurant_name":
            return fake.company()
        elif param == "stock_symbol":
            return fake.lexify(text="???").upper()
        elif param == "recipe_ingredients":
            return [fake.word() for _ in range(random.randint(3, 10))]
        elif param == "transaction_id":
            return fake.uuid4()
        elif param in ["ingredient_list", "task_list"]:
            return [fake.word() for _ in range(random.randint(5, 15))]
        elif param == "destination":
            return fake.address()
        elif param == "position" or param == "window_position":
            return random.choice(["up", "down", "forward", "backward"])
        elif param == "window_location":
            return random.choice(["driver", "passenger", "rear_right", "rear_left"])
        elif param == "method":
            return random.choice(["remote", "keyless", "keyed"])
        elif param == "lock":
            return random.choice(['lock', 'unlock'])
        elif param == "state" or param == "enabled":
            return random.choice(["on", "off"])
        elif param == "zone":
            return random.choice(["front", "rear", "all"])
        elif param == "color":
            return fake.color_name()
        elif param == "tire_id":
            return random.choice(["front_left", "front_right", "rear_left", "rear_right", "all"])
        elif param == "schedule_oil_change":
            return random.choice(["schedule", "start immediately"])
        elif param == "action":
            return random.choice(["activate", "deactivate"])
        elif data_type == "number":
            return random.randint(1, 100)
        elif data_type == "integer":
            return random.randint(1, 100)
        elif data_type == "string":
            return fake.word()
        elif data_type == "array":
            return [fake.word() for _ in range(random.randint(5, 15))]
        else:
            return -1

    def generate_function_call_message(self):
        """
        Generates messages for function calls.

        :return: A list of messages.
        """
        messages = []
        previous_function = None

        for function in self.functions:
            name = function['name']
            description = function['description']
            parameters = function['parameters']['properties']
            required = function['parameters'].get('required', [])
            optional = function['parameters'].get('optional', [])

            # Create a scenario message
            if self.prompt_insertion:
                system_message = "<|im_start|>system\nYou are a helpful assistant with access to the following functions. Use these functions when they are relevant to assist with a user's request\n"
                system_message += "[{\n"
                system_message += f'    "name": "{name}",\n'
                system_message += f'    "description": "{description}",\n'
                system_message += '    "parameters": {\n'
                system_message += '        "type": "object",\n'
                system_message += '        "properties": {\n'

                for param, param_details in parameters.items():
                    system_message += f'            "{param}": {{\n'
                    system_message += f'                "type": "{param_details["type"]}",\n'
                    system_message += f'                "description": "{param_details["description"]}"\n'
                    system_message += '            },\n'

                system_message += '        }\n'
                system_message += '    }\n'
                system_message += '}]<|im_end|>\n'
            else:
                system_message = "<|im_start|>system\nYou are a helpful assistant. You have to either provide a way to answer user's request or answer user's query.\n<|im_end|>\n"

            if not self.read_data_from_file:
                for _ in range(random.randint(40, 100)):
                    user_question = self.get_random_question(name)
                    placeholders = re.findall(r'\{(.*?)\}', user_question)

                    function_args = {}
                    for param, param_details in parameters.items():
                        param_type = param_details['type']
                        if param in placeholders:
                            function_args[param] = self.get_random_value(param, param_type)
                        else:
                            # If parameter not in placeholders, assign default or skip
                            continue

                    user_message = "<|im_start|>user\n"
                    user_message += user_question.format(**function_args)
                    user_message += "<|im_end|>\n"

                    assistant_message = f'<|im_start|>assistant\n<functioncall> {{"name": "{name}", "arguments": "{function_args}"}} <|im_end|><|endoftext|>'

                    scenario_message = {'system': system_message, 'user': user_message, 'assistant': assistant_message}
                    messages.append(scenario_message)
            else:
                if self.data_file is None:
                    raise ValueError("Data file is required to read data from file")

                with open(self.data_file, "r") as file:
                    data = json.load(file)

                for fn in data:
                    user_commands = data[fn]['user_commands']
                    function_calls = data[fn]['calls']
                    fn_args_list = data[fn]['args'][0]  # Access the list of arguments
                    for i in range(len(function_calls)):
                        for j in range(len(user_commands[i])):
                            user_message = "<|im_start|>user\n"
                            user_command_temp = user_commands[i][j]

                            # Apply random modifications to the user command
                            if "please" in user_command_temp and random.random() < 0.8:
                                user_command_temp = user_command_temp.replace("please", "")

                            if "can you " in user_command_temp:
                                user_command_temp = user_command_temp.replace("can you", "")

                            if "car " in user_command_temp and random.random() < 0.5:
                                user_command_temp = user_command_temp.replace("car", "")

                            user_message += user_command_temp
                            user_message += "<|im_end|>\n"

                            assistant_message = f'<|im_start|>assistant\n<functioncall> {{"name": "{fn}", "arguments": "{fn_args_list[i]}"}} <|im_end|><|endoftext|>'
                            scenario_message = {'system': system_message, 'user': user_message, 'assistant': assistant_message}
                            messages.append(scenario_message)

            # Add negative samples with questions not related to the function
            if self.prompt_insertion and previous_function:
                previous_name = previous_function['name']
                previous_parameters = previous_function['parameters']['properties']

                for _ in range(random.randint(40, 100)):
                    user_message = "<|im_start|>user\n"
                    function_args = {}
                    for param, param_details in previous_parameters.items():
                        param_type = param_details['type']
                        function_args[param] = self.get_random_value(param, param_type)

                    user_question = self.get_random_question(previous_name)
                    user_message += user_question.format(**function_args)
                    user_message += "<|im_end|>\n"

                    assistant_message = f"<|im_start|>assistant\nI'm sorry, but I don't have the capability to answer your request. My current function allows me to {description.lower().split('.')[0]}<|im_end|><|endoftext|>"

                    scenario_message = {'system': system_message, 'user': user_message, 'assistant': assistant_message}
                    messages.append(scenario_message)

            previous_function = function

        return messages

    def shuffle_and_split_messages(self, messages):
        """
        Shuffles and splits the messages into training and testing datasets.

        :param messages: List of messages.
        :return: A tuple containing training and testing datasets.
        """
        random.shuffle(messages)
        split_index = int(len(messages) * 0.8)
        train_data = messages[:split_index]
        test_data = messages[split_index:]
        return train_data, test_data

    def save_messages(self, train_data, test_data):
        """
        Saves the training and testing datasets to files.

        :param train_data: List of training messages.
        :param test_data: List of testing messages.
        """
        base_output_file = f"./data/{self.output_file}_{self.prompt_insertion}"
        # Save training data
        with open(f"{base_output_file}-train.json", "w") as file:
            json.dump(train_data, file, indent=4)

        # Save testing data
        with open(f"{base_output_file}-test.json", "w") as file:
            json.dump(test_data, file, indent=4)

    def run(self):
        """
        Executes the full pipeline: generate messages, shuffle, split, and save.
        """
        messages = self.generate_function_call_message()
        train_data, test_data = self.shuffle_and_split_messages(messages)
        self.save_messages(train_data, test_data)

def main():
    parser = argparse.ArgumentParser(description='Generate training data for function calling')
    parser.add_argument('--prompt_insertion', action='store_true', help='Insert prompt for each function')
    parser.add_argument('--no_prompt_insertion', action='store_false', dest='prompt_insertion', help='Do not insert prompt for each function')
    parser.set_defaults(prompt_insertion=False)
    parser.add_argument('--output_file', type=str, default='car_finetuning_gpt', help='Base name for the output files')
    parser.add_argument('--read_data_from_file', action='store_true', help='Read data from file')
    parser.add_argument('--data_file', type=str, default='./data/function_calls_with_commands.json', help='Path to data file if reading from file')
    args = parser.parse_args()
    
    # import pdb; pdb.set_trace()
    
    generator = TrainingDataGenerator(
        prompt_insertion=args.prompt_insertion,
        output_file=args.output_file,
        read_data_from_file=args.read_data_from_file,
        data_file=args.data_file
    )
    generator.run()

if __name__ == "__main__":
    main()