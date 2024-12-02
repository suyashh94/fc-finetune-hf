import os
import json
import random
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import functions
from dotenv import load_dotenv

class FunctionCallGenerator:
    def __init__(self, functions, n=10, opt_prob=0.5, output_file="./data/function_calls.json"):
        """
        Initializes the FunctionCallGenerator with the given parameters.

        :param functions: List of function details from the config.
        :param n: Number of function calls to generate per function.
        :param opt_prob: Probability of including optional parameters.
        :param output_file: Path to the output JSON file.
        """
        self.functions = functions
        self.n = n
        self.opt_prob = opt_prob
        self.output_file = output_file
        self.function_args = {}
        self.function_calls = {}
        load_dotenv()

    def get_param_choices(self, parameters, param_list):
        """
        Generates possible values for each parameter based on its type and constraints.

        :param parameters: Dictionary of parameter details.
        :param param_list: List of parameter names to process.
        :return: Dictionary of parameter choices.
        """
        param_choices = {}
        for param in param_list:
            param_details = parameters[param]
            param_type = param_details["type"]
            param_choices[param] = {'type': param_type}
            if 'enum' in param_details:
                param_choices[param]['values'] = param_details['enum']
            elif param_type == "boolean":
                param_choices[param]['values'] = [True, False]
            elif param_type in ["integer", "number"]:
                lower_bound = param_details.get("lower_bound", 1)
                upper_bound = param_details.get("upper_bound", 100)
                param_choices[param]['values'] = list(range(lower_bound, upper_bound))
            elif param_type == "string":
                param_choices[param]['values'] = ["|string1|", "|string2|", "|string3|"]
            elif param_type == "array":
                items = param_details.get('items', {})
                item_type = items.get('type', 'string')
                param_choices[param]['item_type'] = item_type
                if 'enum' in items:
                    param_choices[param]['values'] = items['enum']
                elif item_type == "string":
                    param_choices[param]['values'] = ["|string1|", "|string2|", "|string3|"]
                elif item_type in ["integer", "number"]:
                    lower_bound = items.get("lower_bound", 1)
                    upper_bound = items.get("upper_bound", 100)
                    param_choices[param]['values'] = list(range(lower_bound, upper_bound))
                elif item_type == "boolean":
                    param_choices[param]['values'] = [True, False]
        return param_choices

    def construct_function(self, required_params_config, opt_params_config):
        """
        Constructs a single function call with randomly selected arguments.

        :param required_params_config: Choices for required parameters.
        :param opt_params_config: Choices for optional parameters.
        :return: Dictionary of function arguments.
        """
        function_args = {}
        for param, choices in required_params_config.items():
            if choices['type'] != 'array':
                function_args[param] = random.choice(choices['values'])
            else:
                array_length = random.randint(1, len(choices['values']))
                function_args[param] = random.sample(choices['values'], array_length)
                if 'all' in function_args[param] and len(function_args[param]) > 1:
                    function_args[param].remove('all')

        for param, choices in opt_params_config.items():
            if random.random() < self.opt_prob:
                if 'array' not in choices['type']:
                    function_args[param] = random.choice(choices['values'])
                else:
                    array_length = random.randint(1, len(choices['values']))
                    function_args[param] = random.sample(choices['values'], array_length)
                    if 'all' in function_args[param] and len(function_args[param]) > 1:
                        function_args[param].remove('all')

        return function_args

    def get_function_params(self, function_detail):
        """
        Extracts the function name and parameter choices from the function details.

        :param function_detail: Dictionary containing function details.
        :return: Tuple containing function name, required parameter choices, and optional parameter choices.
        """
        function_name = function_detail["name"]
        function_parameters = function_detail["parameters"]["properties"]
        required_parameters = function_detail["parameters"].get("required", [])
        optional_parameters = function_detail["parameters"].get("optional", [])

        function_args_choices_req = self.get_param_choices(function_parameters, required_parameters)
        function_args_choices_opt = self.get_param_choices(function_parameters, optional_parameters)

        return function_name, function_args_choices_req, function_args_choices_opt

    def generate_function_call_args(self):
        """
        Generates function call arguments for each function.

        Updates self.function_args with generated arguments.
        """
        function_args = {}
        for function_detail in self.functions:
            function_name, required_params_config, opt_params_config = self.get_function_params(function_detail)
            function_args[function_name] = []
            for _ in range(self.n):
                args = self.construct_function(required_params_config, opt_params_config)
                function_args[function_name].append(args)
        self.function_args = function_args

    def assimilate_function_calls(self):
        """
        Converts the function arguments into readable function call strings.

        Updates self.function_calls with the assimilated calls.
        """
        function_call_main_dict = {}
        for function_name, function_args_list in self.function_args.items():
            function_call_main_dict[function_name] = {"args": function_args_list, "calls": []}
            for args in function_args_list:
                function_call = f"{function_name}("
                params = []
                for param, value in args.items():
                    # Format arrays properly
                    if isinstance(value, list):
                        value_str = "[" + ", ".join(map(str, value)) + "]"
                    else:
                        value_str = str(value)
                    params.append(f"{param}={value_str}")
                function_call += ", ".join(params) + ")"
                function_call_main_dict[function_name]['calls'].append(function_call)
        self.function_calls = function_call_main_dict

    def save_function_calls(self):
        """
        Saves the generated function calls to a JSON file specified by self.output_file.
        """
        currDir = os.path.dirname(os.path.realpath(__file__))
        # find parent of current directory
        parentDir = os.path.abspath(os.path.join(currDir, os.pardir))
        saveDir = os.path.join(parentDir, "data")
        
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        
        save_path = os.path.join(saveDir, self.output_file)
        
        
        with open(save_path, "w") as f:
            json.dump(self.function_calls, f, indent=4)

    def run(self):
        """
        Executes the full pipeline: generating arguments, assimilating calls, and saving to a file.
        """
        self.generate_function_call_args()
        self.assimilate_function_calls()
        self.save_function_calls()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10, help='Number of function calls per function')
    parser.add_argument('--opt_prob', type=float, default=0.5, help='Probability of including optional parameters')
    parser.add_argument('--output_file', type=str, default='function_calls.json', help='Output file path')
    args = parser.parse_args()

    generator = FunctionCallGenerator(functions, n=args.n, opt_prob=args.opt_prob, output_file=args.output_file)
    generator.run()

if __name__ == "__main__":
    main()