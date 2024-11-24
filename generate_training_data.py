import os
import random
from datetime import datetime, timedelta
from faker import Faker
import numpy as np
import re
from faker import Faker
from datetime import datetime, timedelta
import random


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
    
def generate_function_call_message(functions: list, questions_dict: dict[str: list[str]], prompt_insertion = True) -> list:
    
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

functions = [
    {
        "name": "adjust_temperature",
        "description": "Adjust the temperature in a specified zone of the car.",
        "parameters": {
            "type": "object",
            "properties": {
                "zone": {
                    "type": "string",
                    "enum": ["front", "rear", "all"],
                    "description": "The zone where the temperature will be adjusted.",
                    "default": "all"
                },
                "temperature": {
                    "type": "number",
                    "description": "The target temperature for the specified zone in degrees Celsius."
                }
            },
            "required": ["temperature"],
            "optional": ["zone"]
        }
    },
    {
        "name": "adjust_seat",
        "description": "Adjust a seat's position in the car.",
        "parameters": {
            "type": "object",
            "properties": {
                "seat_type": {
                    "type": "string",
                    "enum": ["driver", "passenger"],
                    "description": "The type of seat to adjust.",
                    "default": "driver"
                },
                "position": {
                    "type": "string",
                    "description": "The desired position of the seat (e.g., 'forward', 'backward', 'up', 'down')."
                },
            },
            "required": ["position"],
            "optional": ["seat_type"]
        }
    },
    {
        "name": "control_window",
        "description": "Control the car window's position.",
        "parameters": {
            "type": "object",
            "properties": {
                "window_position": {
                    "type": "string",
                    "description": "The desired position of the window (e.g., 'up', 'down')."
                },
                "window_location": {
                    "type": "string",
                    "description": "The location of the window (e.g., 'driver', 'passenger', 'rear_right', 'rear_left').",
                    "default": "driver"
                }
            },
            "required": ["window_position"],
            "optional": ["window_location"]
        }
    },
    {
        "name": "adjust_wiper_speed",
        "description": "Activate the windshield wipers.",
        "parameters": {
            "type": "object",
            "properties": {
                "speed": {
                    "type": "integer",
                    "description": "The speed of the wipers (e.g., 1 for low, 2 for medium, 3 for high)."
                }
            },
            "required": [
                "speed"
            ]
        }
    },
    {
        "name": "activate_defroster",
        "description": "Activate the defroster for windows and windshield.",
        "parameters": {
            "type": "object",
            "properties": {
                "defroster_zone": {
                    "type": "string",
                    "description": "The zone to defrost (e.g., 'front', 'rear', 'all').",
                    "default": "all"
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Duration in minutes for which the defroster should be active.",
                    "default": 10
                }
            },
            "required": [],
            "optional": [
                "duration_minutes", "defroster_zone"
            ]
        }
    },
    {
        "name": "start_engine",
        "description": "Start the car's engine remotely.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "The method to start the engine (e.g., 'remote', 'keyless', 'keyed').",
                    "default": "keyless"
                }
            },
            "optional": [
                "method"
            ]
        }
    },
    {
        "name": "lock_doors",
        "description": "Lock or unlock the car doors.",
        "parameters": {
            "type": "object",
            "properties": {
                "lock": {
                    "type": "string",
                    "description": "Set to true to lock the doors, false to unlock."
                }
            },
            "required": [
                "lock"
            ]
        }
    },
    {
        "name": "play_music",
        "description": "Control the music player in the car.",
        "parameters": {
            "type": "object",
            "properties": {
                "track": {
                    "type": "string",
                    "description": "The track name to play.",
                    "default": "random"
                },
                "volume": {
                    "type": "integer",
                    "description": "Volume level from 1 (low) to 10 (high).",
                    "default": 5
                }
            },
            "required": [],
            "optional": [
                "volume","track"
            ]
        }
    },
    {
        "name": "toggle_headlights",
        "description": "Turn the headlights on or off.",
        "parameters": {
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "Set to true to turn the headlights on, false to turn them off."
                }
            },
            "required": [
                "state"
            ]
        }
    },
    {
        "name": "set_navigation_destination",
        "description": "Set a destination in the car's navigation system.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "The address or location to navigate to."
                }
            },
            "required": [
                "destination"
            ]
        }
    },
    {
        "name": "control_ambient_lighting",
        "description": "Adjust the color and intensity of the interior ambient lighting.",
        "parameters": {
            "type": "object",
            "properties": {
                "color": {
                    "type": "string",
                    "description": "The color of the ambient lighting."
                },
                "intensity": {
                    "type": "integer",
                    "description": "The intensity level of the lighting, from 1 (low) to 10 (high).",
                    "default": 5
                }
            },
            "required": [
                "color"
            ],
            "optional": [
                "intensity"
            ]
        }
    },
    {
        "name": "set_cruise_control",
        "description": "Activate and set the speed for cruise control.",
        "parameters": {
            "type": "object",
            "properties": {
                "speed": {
                    "type": "integer",
                    "description": "The cruise control speed in km/h."
                }
            },
            "required": [
                "speed"
            ]
        }
    },
    {
        "name": "check_battery_health",
        "description": "Provide the current status and health of the car's battery.",
        "parameters": {
            "type": "object",
            "properties": {
                "include_history": {
                    "type": "boolean",
                    "description": "Whether to include historical health data.",
                    "default": False
                }
            },
            "optional": [
                "include_history"
            ]
        }
    },
    {
        "name": "toggle_sport_mode",
        "description": "Toggle the car's sport mode setting.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Set to true to enable sport mode, false to disable."
                }
            },
            "required": [
                "action"
            ]
        }
    }

]

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
    parser.set_defaults(prompt_insertion=True)
    parser.add_argument('--output_file', type=str, default='car_finetuning', help='Output file path')
    
    args = parser.parse_args()
    prompt_insertion = args.prompt_insertion
    
    output_file = args.output_file
    output_file = f"./data/{output_file}_{prompt_insertion}"
    

    # Generate messages
    messages = generate_function_call_message(functions, questions_dict, prompt_insertion=prompt_insertion)

    # save the messages
    np.save(f"{output_file}.npy", messages)

    
    with open(f"{output_file}.json", "w") as file:
        json.dump(messages, file, indent=4)
        
    messages = np.load(f'{output_file}.npy', allow_pickle=True)

    from collections import Counter

    function_counter = Counter()

    # Iterate through the records and extract function names where applicable
    for record in messages:
        assistant_response = record['assistant']
        if '<functioncall>' in assistant_response:
            # Extract the function name
            function_name = assistant_response.split('"name": ')[1].split(',')[0].replace('"', '').strip()
            function_counter[function_name] += 1
        else:
            function_counter['Error'] += 1

    # Convert to a dictionary for easier use in visualization
    function_data = dict(function_counter)