import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import random
from datetime import datetime, timedelta
from faker import Faker
import numpy as np
import re
from faker import Faker
from datetime import datetime, timedelta
import random
import json
from config import functions
import inspect

import ast

def extract_function_name_and_parameters(function_call):
    """
    Extracts the function name and parameter names and their values from a function call string.

    Args:
        function_call (str): The function call string.

    Returns:
        dict: A dictionary containing the function name and a dictionary of parameter names and values.
    """
    def resolve_value(node):
        """
        Resolves an AST node to its Python representation.

        Args:
            node (ast.AST): The AST node to resolve.

        Returns:
            The Python representation of the node, or the string representation for unsupported cases.
        """
        if isinstance(node, ast.Constant):  # Handles literals like 19, "string", True
            return node.value
        elif isinstance(node, ast.List):  # Handles lists
            return [resolve_value(element) for element in node.elts]
        elif isinstance(node, ast.Name):  # Handles variable names like 'a', 'b'
            return node.id
        elif isinstance(node, ast.Tuple):  # Handles tuples
            return tuple(resolve_value(element) for element in node.elts)
        elif isinstance(node, ast.BinOp):  # Handles binary operations (e.g., rear-left)
            left = resolve_value(node.left)
            op = resolve_operator(node.op)
            right = resolve_value(node.right)
            return f"{left}{op}{right}"
        else:
            return str(node)  # Fallback to string representation

    def resolve_operator(op):
        """
        Resolves an AST operator node to its string representation.

        Args:
            op (ast.operator): The AST operator node.

        Returns:
            str: The string representation of the operator.
        """
        if isinstance(op, ast.Sub):
            return '-'
        # Add other operators if needed
        return ""

    # Parse the function call string into an AST
    try:
        parsed = ast.parse(function_call.strip(), mode='eval')
    except SyntaxError as e:
        return {"error": f"Invalid function call: {e}"}

    # Ensure the parsed tree has a Call node
    if not isinstance(parsed.body, ast.Call):
        return {"error": "Input is not a valid function call"}

    # Extract the function name
    function_name = parsed.body.func.id if isinstance(parsed.body.func, ast.Name) else str(parsed.body.func)

    # Extract arguments
    params = {}
    for keyword in parsed.body.keywords:
        key = keyword.arg
        value = resolve_value(keyword.value)
        params[key] = value

    # Return the function name and parameters
    return {
        "function_name": function_name,
        "parameters": params
    }

    
def generate_function_call_message(functions: list,data_file = None, include_incomplete = False) -> list:
    
    # message initialization
    complete_messages = []
    incomplete_messages = []
    previous_function = None

    
    # name = function['name']
    # description = function['description']
    # parameters = function['parameters']['properties']
    # required = function['parameters'].get('required', [])
    # optional = function['parameters'].get('optional', [])
        
        
       
    system_message = "<|im_start|>system\nYou are a helpful assistant. You have to either provide a way to answer user's request or answer user's query.\n<|im_end|>\n"
    
    
    if data_file is None:
        raise ValueError("Data file is required to read data from file")
    
    with open(data_file, "r") as file:
        data = json.load(file)
    
    for fn in data:
        user_commands = data[fn]['complete_commands']
        function_calls = data[fn]['calls']
        fn_args = data[fn]['args']
        # import pdb; pdb.set_trace()
        
        fun = list(filter(lambda x: x['name'] == fn, functions))
        fun = fun[0]
        required = fun['parameters'].get('required', [])
        optional = fun['parameters'].get('optional', [])
        
        # import pdb; pdb.set_trace()
        
        for i in range(len(user_commands)):
            for j in range(len(user_commands[i])):
                # import pdb; pdb.set_trace()
                user_message = "<|im_start|>user\n"
                user_command_temp = user_commands[i][j]
                user_command_temp = refine_command_expression(user_command_temp)
                user_message += user_commands[i][j]
                user_message += "<|im_end|>\n"
                assistant_message = f'<|im_start|>assistant\n<functioncall> {{"name": "{fn}", "arguments": "{fn_args[i]}"}} <|im_end|><|endoftext|>'
                scenario_message = {'system': system_message, 'user': user_message, 'assistant': assistant_message}
                complete_messages.append(scenario_message)
        
        ## Continue if function has no required parameters as the info of calling 
                ## the function with no required parameters is already present in the complete commands
                ## so model would have learnt it from there
        if len(required) == 0:
            continue
        
        if include_incomplete:
            incomplete_commands = data[fn]['incomplete_commands']
            for i in range(len(incomplete_commands)):
                for j in range(len(incomplete_commands[i])):
                    
                    if len(incomplete_commands[i][j]['incomplete_command']) == 0:
                        continue
                    
                    user_message = "<|im_start|>user\n"
                    user_command_temp = incomplete_commands[i][j]['incomplete_command']
                    user_message += user_command_temp
                    user_message += "<|im_end|>\n"
                    
                    fn_call_temp = incomplete_commands[i][j]['modified_incorrect_function_call']
                    fn_args_temp = extract_function_name_and_parameters(fn_call_temp)
                    
                    
                    
                    ## Check if no args are present and if so, add POSSIBILY_INCORRECT to function name
                    try:
                        if len(fn_args_temp['parameters']) == 0:
                            fn_args_temp['function_name'] = "POSSIBLY_INCORRECT_" + fn_args_temp['function_name']
                            print(f"Possibly incorrect example: {incomplete_commands[i][j]['incomplete_command']} \n , fn_args : {fn_args_temp}" )
                    except:
                        print(f"Error in extracting function name and parameters for {fn_call_temp}")
                        continue
                    
                    assistant_message = f'<|im_start|>assistant\n<functioncall> {{"name": "{fn_args_temp["function_name"]}", "arguments": "{fn_args_temp["parameters"]}"}} <|im_end|><|endoftext|>'
                    scenario_message = {'system': system_message, 'user': user_message, 'assistant': assistant_message}
                    incomplete_messages.append(scenario_message)
                
                            
    return complete_messages, incomplete_messages

def refine_command_expression(user_command_temp):
    if "please" in user_command_temp and random.random() < 0.8: 
        user_command_temp = user_command_temp.replace("please", "")
                    
                    # strip "can you" if it is present
    if "can you " in user_command_temp:
        user_command_temp = user_command_temp.replace("can you", "")
                    
                    # strip "car" if it is present with probability 0.5
    if "car " in user_command_temp and random.random() < 0.5:
        user_command_temp = user_command_temp.replace("car", "")
    
    return user_command_temp

def save_data(complete_messages, output_file, suffix = ''):
        # split the data into train and test
        split = int(len(complete_messages) * 0.8)
        train_data = complete_messages[:split]
        test_data = complete_messages[split:]
        # save the messages
        np.save(f"{output_file}-train_{suffix}.npy", train_data)

        with open(f"{output_file}-train_{suffix}.json", "w") as file:
            json.dump(train_data, file, indent=4)
            
        np.save(f"{output_file}-test_{suffix}.npy", test_data)
        with open(f"{output_file}-test_{suffix}.json", "w") as file:
            json.dump(test_data, file, indent=4)

if __name__ == "__main__":
    
    import argparse
    import json
    
    # read arguments
    parser = argparse.ArgumentParser(description='Generate training data for function calling')
    parser.add_argument('--output_file', type=str, default='car_finetuning_gpt', help='Output file path')
    parser.add_argument('--data_file', type=str, default='./data/function_calls_with_commands.json', help='Data file path')
    parser.add_argument('--include_incomplete', type=bool, default=False, help='Include incomplete function calls')
    
    args = parser.parse_args()
    
    output_file = args.output_file
    
    # import pdb; pdb.set_trace()
    
    complete_messages, incomplete_messages = generate_function_call_message(functions, data_file=args.data_file, include_incomplete=args.include_incomplete)
    random.shuffle(complete_messages)
    random.shuffle(incomplete_messages)
    
    save_data(complete_messages, output_file, suffix='complete')
    save_data(incomplete_messages, output_file, suffix='incomplete')
    
    