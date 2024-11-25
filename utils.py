import json

def convert_command(command):
    try:
        # Manually extract the JSON portion between known delimiters
        start_delim = '<functioncall> '
        end_delim = '<|im_end|>'
        start_idx = command.find(start_delim) + len(start_delim)
        end_idx = command.find(end_delim)

        # import pdb; pdb.set_trace()
        
        if start_idx == -1 or end_idx == -1:
            return {
                "error": "Delimiters not found"
            }

        # Extract the JSON string
        json_str = command[start_idx:end_idx]

        # Load the JSON string into a dictionary
        function_call_dict = json.loads(json_str)

        # Extract function name and arguments
        fn_name = function_call_dict.get('name')
        arguments = function_call_dict.get('arguments', '{}')

        # import pdb; pdb.set_trace()
        
        # Load arguments JSON, taking care of the replacing if necessary
        arguments = arguments.replace("'", '"')
        # replace True with 'true' and False with 'false' if they are present in the arguments
        arguments = arguments.replace('True', 'true')
        arguments = arguments.replace('False', 'false')
        
        properties = json.loads(arguments)

        # Construct the final dictionary
        result = {
            'fn_name': fn_name,
            'properties': properties
        }

        return result

    except Exception as e:
        return {
            "error": str(e)
        }
    
if __name__ == '__main__':
    # Example command input, directly copying your provided debug output
    command = "<|im_start|>assistant\n<functioncall> {\"name\": \"adjust_temperature\", \"arguments\": \"{'temperature': 14}\"} <|im_end|><|endoftext|>"
    command = "<|im_start|>assistant\n<functioncall> {\"name\": \"check_battery_health\", \"arguments\": \"{'include_history': True}\"} <|im_end|><|endoftext|>"
    converted_command = convert_command(command)
    print(converted_command)