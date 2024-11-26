import os
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

def get_randon_question(function_name:str, questions_dict: dict[str: list[str]]) ->str:
    question_list = questions_dict.get(function_name)
    if question_list:
        return random.choice(question_list)
    else:
        return ""  



def get_random_value(param: str, data_type: str):
    fake = Faker()

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
    
def generate_function_call_message(functions: list, questions_dict: dict[str: list[str]], prompt_insertion = True, read_data_from_file = False, data_file = None) -> list:
    
    # message initialization
    messages = []
    previous_function = None

    for function in functions:

        name = function['name']
        description = function['description']
        parameters = function['parameters']['properties']
        required = function['parameters'].get('required', [])
        optional = function['parameters'].get('optional', [])
        
        
        # Create a scenario message
        if prompt_insertion:
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
            
            system_message += '        },\n'
            system_message += f'        "required": {required}\n'
            system_message += '    }\n'
            system_message += '}]<|im_end|>\n'
        else:
            system_message = "<|im_start|>system\nYou are a helpful assistant. You have to either provide a way to answer user's request or answer user's query.\n<|im_end|>\n"
        
        if not read_data_from_file:
            for i in range(random.randint(40,100)):
                
                ## BUG HERE RIGHT NOW
                user_question = get_randon_question(name, questions_dict)
                # import pdb; pdb.set_trace()
                placeholders = re.findall(r'\{(.*?)\}', user_question)
                
                #TODO add required placeholder check else wise negative sample with incomplete info 
                
                
                
                user_message = "<|im_start|>user\n"
                function_args = {}
                # for param in required:
                for param, param_details in parameters.items():
                    param_type = param_details['type']
                    if param in placeholders:
                        function_args[param] = get_random_value(param, param_type)
                    else:
                        function_args[param] = param_details.get('default',[])

                
                
                user_message += user_question.format(**function_args)
                user_message += "<|im_end|>\n"
                        
                assistant_message = f'<|im_start|>assistant\n<functioncall> {{"name": "{name}", "arguments": "{function_args}"}} <|im_end|><|endoftext|>'
                
                scenario_message = {'system': system_message, 'user': user_message, 'assistant': assistant_message}
                messages.append(scenario_message)
        else:
            if data_file is None:
                raise ValueError("Data file is required to read data from file")
            
            with open(data_file, "r") as file:
                data = json.load(file)
            
            for fn in data:
                user_commands = data[fn]['user_commands']
                function_calls = data[fn]['calls']
                fn_args = data[fn]['args']
                for i in range(len(function_calls)):
                    for j in range(len(user_commands[i])):
                        # import pdb; pdb.set_trace()
                        user_message = "<|im_start|>user\n"
                        user_command_temp = user_commands[i][j]
                        # strip 'please' if it is present with probability 0.8 
                        if "please" in user_command_temp and random.random() < 0.8: 
                            user_command_temp = user_command_temp.replace("please", "")
                        
                        # strip "can you" if it is present
                        if "can you " in user_command_temp:
                            user_command_temp = user_command_temp.replace("can you", "")
                        
                        # strip "car" if it is present with probability 0.5
                        if "car " in user_command_temp and random.random() < 0.5:
                            user_command_temp = user_command_temp.replace("car", "")
                        
                        user_message += user_commands[i][j]
                        user_message += "<|im_end|>\n"
                        assistant_message = f'<|im_start|>assistant\n<functioncall> {{"name": "{fn}", "arguments": "{fn_args[0][i]}"}} <|im_end|><|endoftext|>'
                        scenario_message = {'system': system_message, 'user': user_message, 'assistant': assistant_message}
                        messages.append(scenario_message)
                        
                        
        
        # Add questions not related with the function
        if prompt_insertion:
            if previous_function:
                previous_name = previous_function['name']
                previous_parameters = previous_function['parameters']['properties']
                
                for i in range(random.randint(40, 100)):

                    user_message = "<|im_start|>user\n"
                    function_args = {}
                    for param, param_details in previous_parameters.items():
                        param_type = param_details['type']
                        function_args[param] = get_random_value(param, param_type)

                    user_message += get_randon_question(previous_name, questions_dict).format(**function_args)
                    user_message += "<|im_end|>\n"
                    
                    assistant_message = f"<|im_start|>assistant\nI'm sorry, but I don't have the capability to answer your request. My current function allows me to {description.lower().split('.')[0]}<|im_end|><|endoftext|>"
                    
                    scenario_message = {'system': system_message, 'user': user_message, 'assistant': assistant_message}
                    
                    messages.append(scenario_message)
            previous_function = function
    return messages



questions_dict = {
        "adjust_temperature": [
            "Can you set the {zone} temperature to {temperature} degrees?",
            "Please adjust the {zone} zone temperature to {temperature}.",
            "I need the {zone} area to be {temperature} degrees.",
            "Set the {zone} temperature to {temperature} degrees Celsius.",
            "Adjust the {zone} zone to {temperature} degrees, please.",
            "Adjust temperature to {temperature}",
            "Set temperature to {temperature}"
        ],
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
        
        "control_window": [
            "Can you roll the {window_location} window {window_position}?",
            "Please {window_position} the {window_location} window.",
            "I need the {window_location} window {window_position}.",
            "Set the {window_location} window to {window_position}.",
            "Move the {window_location} window {window_position}."
            "Roll the window {window_position}",
            "Move the window {window_position}",
            "I need the window {window_position}"
        ],
        
        "adjust_wiper_speed": [
            "Can you set the wipers to speed {speed}?",
            "Please activate the wipers at speed {speed}.",
            "I need the windshield wipers on speed {speed}.",
            "Turn on the wipers at speed {speed}.",
            "Activate the wipers at speed {speed}"
        ],
        "activate_defroster": [
            "Can you activate the {defroster_zone} defroster for {duration_minutes} minutes?",
            "Please turn on the {defroster_zone} defroster for {duration_minutes} minutes.",
            "I need the {defroster_zone} defroster on for {duration_minutes} minutes.",
            "Set the {defroster_zone} defroster to run for {duration_minutes} minutes.",
            "Defrost the {defroster_zone}, duration: {duration_minutes} minutes.",
            "Defrost {defroster_zone} part",
            "Turn on the defroster for {duration_minutes} minutes",
            "Activate defroster",
            "Turn on defroster",
            "I cannot see, turn on defroster",
            "It is all foggy, help me see!"
        ],
        "start_engine": [
            "Start the engine",
            "Turn on the engine",
            "Can you start the engine using the {method} method?",
            "Please start the engine",
            "I need the engine started by {method}.",
            "Activate the engine start.",
            "Turn on the engine using {method}."
        ],
        "lock_doors": [
            "Can you {lock} the doors?",
            "Please {lock} all the car doors.",
            "I need the doors {lock}.",
            "Set the door locks to {lock}.",
            "Lock the doors: {lock}."
        ],
        
        "play_music": [
            "Can you play the track '{track}' at volume {volume}?",
            "Please play '{track}' with the volume set to {volume}.",
            "I want to listen to '{track}'",
            "Turn on the music '{track}'",
            "Start playing '{track}' at {volume} volume.",
            "Play some music!",
            "Turn on the music player",
            "Play the song '{track}'",
        ],
        "toggle_headlights": [
            "Can you turn the headlights {state}?",
            "Please {state} the headlights.",
            "I need the headlights {state}.",
            "Set the headlights to {state}.",
            "Toggle the headlights: {state}."
        ],
        "set_navigation_destination": [
            "Can you set the navigation destination to {destination}?",
            "Please navigate to {destination}.",
            "I need to go to {destination}.",
            "Set the destination to {destination}.",
            "Navigate to {destination}."
        ],
        "control_ambient_lighting": [
            "Can you set the ambient lighting color to {color}?",
            "Please adjust the ambient lighting to color {color} and intensity {intensity}.",
            "I need the ambient lighting to be color {color} and intensity {intensity}.",
            "Set the ambient lighting to color {color} and intensity {intensity}.",
            "Adjust the ambient lighting to color {color} ."
        ],
        "set_cruise_control": [
            "Can you set the cruise control speed to {speed} km/h?",
            "Please activate cruise control at {speed} km/h.",
            "I need the cruise control set to {speed} km/h.",
            "Set the cruise control speed to {speed} km/h.",
            "Activate cruise control at {speed} km/h."
        ],
        "check_battery_health": [
            "Can you check the battery health?",
            "Please provide the battery health status.",
            "I need to know the battery health.",
            "Check the battery health.",
            "Battery health status."
        ],
        "toggle_sport_mode": [
            "Can you {action} sport mode?",
            "Please {action} sport mode.",
            "I need to {action} sport mode.",
            "{action} sport mode.",
            "{action} sport mode."
        ]
        
    }



if __name__ == "__main__":
    
    import argparse
    import json
    
    # read arguments
    parser = argparse.ArgumentParser(description='Generate training data for function calling')
    parser.add_argument('--prompt_insertion', action='store_true', help='Insert prompt for each function')
    parser.add_argument('--no_prompt_insertion', action='store_false', dest='prompt_insertion', help='Do not insert prompt for each function')
    parser.set_defaults(prompt_insertion=False)
    parser.add_argument('--output_file', type=str, default='car_finetuning_gpt', help='Output file path')
    
    args = parser.parse_args()
    prompt_insertion = args.prompt_insertion
    
    output_file = args.output_file
    output_file = f"./data/{output_file}_{prompt_insertion}"
    

    # Generate messages
    messages = generate_function_call_message(functions, questions_dict, prompt_insertion=prompt_insertion, read_data_from_file=True, data_file="./data/function_calls_with_commands.json")
    # shuffle the messages
    random.shuffle(messages)
    
    ## split the data into train and test
    split = int(len(messages) * 0.8)
    train_data = messages[:split]
    test_data = messages[split:]
    

    # save the messages
    np.save(f"{output_file}-train.npy", train_data)

    with open(f"{output_file}-train.json", "w") as file:
        json.dump(train_data, file, indent=4)
        
    np.save(f"{output_file}-test.npy", test_data)
    with open(f"{output_file}-test.json", "w") as file:
        json.dump(test_data, file, indent=4)