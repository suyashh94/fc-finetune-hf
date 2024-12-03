
from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from enum import Enum

class CommandType(Enum):
    CORRECT_COMMANDS = "correct_commands"
    MISSING_VALUES_COMMANDS = "missing_values_commands"


class CorrectCommandsOutput(BaseModel):
    commands: List[str] = Field(..., description="List of user commands generated for the function call.")

class MissingValuesCommandOutput(BaseModel):
    command : str = Field(..., description="User command generated for the function call with missing values.")
    missing_params: List[str] = Field(..., description="List of parameters with missing value in the command.")

class MissingValuesOutput(BaseModel):
    result: List[MissingValuesCommandOutput] = Field(..., description="List of user commands generated for the function call with missing values.")

class IncompleteCommandOutput(BaseModel):
    incomplete_command: str|None = Field(..., description="Incomplete user command generated for the function call.")
    modified_incorrect_function_call: str|None = Field(..., description="Modified function call with missing values.")

class IncompleteCommands(BaseModel):
    incomplete_commands: List[IncompleteCommandOutput] = Field(..., description="List of incomplete user commands generated for the function call.")

class SampleCorrectnessJudgement(BaseModel):
    judgement: bool = Field(..., description="Judgement of correctness of the modified function call.")
    reason: str = Field(..., description="Reason for the judgement.")

complete_command_gen_prompt = [
                ("system", "You are a helpful AI assistant that generates natural-sounding user commands for a voice-enabled car assistant."),
(
    "human",
    "Your task is to create a set of natural language commands that a user might say to a car assistant, \
    which would cause it to execute the given function call with the specified parameters."
),
(
    "human",
    "Please generate 5 diverse user commands in natural language that correctly correspond to \
    the specified function call: <FUNCTION_CALL>."
),
(
    "human",
    "Ensure the commands are phrased naturally and conversationally, as a user would speak to a car assistant. \
    Use different ways to phrase the command, including variations in tone, style, and word choice."
),
(
    "human",
    "Each command should correctly reflect the parameter values specified in the function call but rephrase \
    the parameter names using synonyms or everyday language. Avoid technical terms or direct references to \
    the function call, ensuring the commands feel intuitive and user-friendly."
),
(
    "human",
    "Here are examples to help you understand how to phrase the commands:"
),
(
    "human",
    "1. For `set_temperature(temperature=35, zone=['driver'])`, commands might include: \
        'Set the temperature to 35 degrees on my side,' 'Make my side warmer to 35 degrees,' or \
        'Turn up the heat to 35 degrees on the driver’s side.'"
),
(
    "human",
    "2. For `adjust_fan_speed(area=['rear-left', 'rear-right'])`, commands could include: \
        'Increase the fan speed in the back seats,' 'Turn up the air flow for rear passengers,' or \
        'Make the fans stronger for the back row.'"
),
(
    "human",
    "3. For `control_window(window_position='close', window_location='passenger')`, commands might be: \
        'Please close the passenger window,' 'Shut the window on the passenger side,' or \
        'Bring the passenger-side window up.'"
),
(
    "human",
    "4. For `activate_defroster(duration_minutes=20, defroster_zone='all')`, commands could be: \
        'Turn on the defroster for 20 minutes everywhere,' 'Please defrost all the windows for 20 minutes,' or \
        'Run the defroster on all windows for 20 minutes.'"
),
(
    "human",
    "5. For `set_wiper_speed(speed='HIGH')`, commands might include: \
        'Set the windshield wipers to high speed,' 'Turn the wipers to their fastest setting,' or \
        'Make the wipers go faster at high speed.'"
),
(
    "human",
    "6. For `lock_doors(lock='lock')`, commands could include: \
        'Lock all the doors,' 'Secure the car doors,' or 'Make sure the car is locked.'"
),
(
    "human",
    "7. For `play_music(volume=5, track='Imagine')`, commands might be: \
        'Play \"Imagine\" at volume 5,' 'Turn up the music to level 5 and play \"Imagine,\"' or \
        'Start playing \"Imagine\" with the sound at 5.'"
),
(
    "human",
    "Remember to be creative and consider different ways users might phrase the same request, \
    including polite, casual, or even commanding tones. Avoid including any technical terms, code, \
    or mention of the function call in the output."
),
(
    "human",
    "Output only the user commands as a list of 5 diverse, natural-sounding sentences, and nothing else."
),
(
    "human",
    "<FUNCTION_CALL> is {function_call}"
)

            ]

incomplete_command_gen_prompt = \
    [
        (
            "system",
            "You are a helpful AI assistant that generates natural-sounding user commands for a voice-enabled car assistant. Your task is to create a set of natural language commands that a user might say, which would cause the car assistant to execute the given function call with some or all parameter values missing. However, the commands should mention the parameter names or their synonyms."
        ),
        (
            "human",
            "Please generate 5 diverse user commands in natural language that would cause the car assistant to execute the following function call with missing parameter values: <FUNCTION_CALL>"
        ),
        (
            "human",
            "Ensure that the commands are phrased naturally, as a user would speak them in conversation with the car assistant."
        ),
        (
            "human",
            "The commands should mention the parameter names (or their synonyms), but may have one, some, or all of the parameter values missing."
        ),
        (
            "human",
            "For example:"
        ),
        (
            "human",
            "1. If the function call is `set_temperature(temperature=25)`, possible commands could be 'Set the temperature please', or 'Adjust the temperature to', where the value '25' is missing."
        ),
        (
            "human",
            "2. If the function call is `adjust_fan_speed(speed='increase')`, possible commands could be 'Adjust the fan speed', or 'Change fan speed' without specifying the information in the function call."
        ),
        (
            "human",
            "3. If the function call is `control_window(window_position='close', window_location='rear_left')`, possible commands could be 'Close the window', 'Control the rear left window', or 'Adjust the window position', where one or both parameter values are missing."
        ),
        (
            "human",
            "If the function call has no parameters, return an empty list."
        ),
        (
            "human",
            "Do not include any technical terms, code, or mention of the function call in your response. Be creative and think of different ways a user might phrase incomplete commands. Users can be polite or abrupt, so consider different tones and styles of speech."
        ),
        (
            "human",
            "For each command, provide the missing parameter names in a list."
        ),
        (
            "human",
            "Output only the list of commands with their missing parameters, and nothing else, in the following format:"
        ),
         (
            "human",
            "Make sure to include all the missing parameters in the output."
        ),
        (
            "human",
            "Function Call is {function_call}"
        ),
    ]


incomplete_command_gen_prompt_reinforced = \
    [
       ("system", "You are a helpful AI assistant that generates natural-sounding user commands for a voice-enabled car assistant."),
("human", "Your task is to analyze a <CORRECT_FUNCTION_CALL> and its corresponding <COMPLETE_COMMAND>, and generate:"),
("human", "1. A <INCOMPLETE_COMMAND>, which is an incomplete version of the <COMPLETE_COMMAND> that omits some or all specific parameter or value information."),
("human", "2. A <MODIFIED_INCORRECT_FUNCTION_CALL>, which is a modified version of the <CORRECT_FUNCTION_CALL> that aligns with the <INCOMPLETE_COMMAND> but is inherently incorrect due to missing or incomplete parameter information."),
("human", "Guidelines for generating <INCOMPLETE_COMMAND> and <MODIFIED_INCORRECT_FUNCTION_CALL>:"),
("human", "1. The <INCOMPLETE_COMMAND> must preserve the specificity of the <COMPLETE_COMMAND>. Generalizing or replacing specific terms (e.g., 'headlights' to 'lights') is NOT allowed."),
("human", "2. The <INCOMPLETE_COMMAND> should only omit specific details about parameters or values from the <COMPLETE_COMMAND>, making it less precise but still logically correct."),
("human", "3. If a parameter or value is missing from the <INCOMPLETE_COMMAND>, the <MODIFIED_INCORRECT_FUNCTION_CALL> MUST NOT include it unless it is explicitly mentioned or can be logically inferred."),
("human", "4. If the <CORRECT_FUNCTION_CALL> has one parameter, the <INCOMPLETE_COMMAND> must remove the information about that parameter entirely."),
("human", "5. If the <CORRECT_FUNCTION_CALL> has multiple parameters, the <INCOMPLETE_COMMAND> must remove information about at least one parameter while keeping the rest, ensuring the command remains incomplete."),
("human", "6. If the <CORRECT_FUNCTION_CALL> has no parameters, both <INCOMPLETE_COMMAND> and <MODIFIED_INCORRECT_FUNCTION_CALL> must be empty strings ('')."),
("human", "8. The <MODIFIED_INCORRECT_FUNCTION_CALL> must reflect only what is explicitly mentioned or reasonably inferable from the <INCOMPLETE_COMMAND>. Any parameter or value not mentioned in the <INCOMPLETE_COMMAND> must be omitted."),
("human", "Definitions:"),
("human", "   - A 'parameter' is any named attribute in the function call (e.g., light_type, temperature, state)."),
("human", "   - A 'value' is the associated value of a parameter (e.g., headlights, 25, on)."),
("human", "   - A '<COMPLETE_COMMAND>' is the natural language input that directly translates into the <CORRECT_FUNCTION_CALL>."),
("human", "   - A '<INCOMPLETE_COMMAND>' is a partial version of the <COMPLETE_COMMAND> that omits some parameter or value information while remaining natural-sounding and specific."),
("human", "   - A '<MODIFIED_INCORRECT_FUNCTION_CALL>' is a version of the <CORRECT_FUNCTION_CALL> altered to match the <INCOMPLETE_COMMAND>, which is inherently incorrect due to missing or incomplete information."),
("human", "Examples:"),
("human", "1. <CORRECT_FUNCTION_CALL>: toggle_lights(light_type='headlights', state='off')"),
("human", "   <COMPLETE_COMMAND>: 'Turn off the headlights.'"),
("human", "   <INCOMPLETE_COMMAND>: 'Turn off.'"),
("human", "   <MODIFIED_INCORRECT_FUNCTION_CALL>: toggle_lights(state='off')"),
("human", "2. <CORRECT_FUNCTION_CALL>: toggle_lights(light_type='interior', state='on')"),
("human", "   <COMPLETE_COMMAND>: 'Turn on the interior lights.'"),
("human", "   <INCOMPLETE_COMMAND>: 'Turn on the lights.'"),
("human", "   <MODIFIED_INCORRECT_FUNCTION_CALL>: toggle_lights(state='on')"),
("human", "3. <CORRECT_FUNCTION_CALL>: start_engine()"),
("human", "   <COMPLETE_COMMAND>: 'Start the car engine.'"),
("human", "   <INCOMPLETE_COMMAND>: '' (empty string)"),
("human", "   <MODIFIED_INCORRECT_FUNCTION_CALL>: '' (empty string)"),
("human", "4. <CORRECT_FUNCTION_CALL>: set_temperature(temperature=25, unit='Celsius')"),
("human", "   <COMPLETE_COMMAND>: 'Set the temperature to 25 degrees Celsius.'"),
("human", "   <INCOMPLETE_COMMAND>: 'Set the temperature.'"),
("human", "   <MODIFIED_INCORRECT_FUNCTION_CALL>: set_temperature()"),
("human", "Your Output:"),
("human", "Output only the <INCOMPLETE_COMMAND> and corresponding <MODIFIED_INCORRECT_FUNCTION_CALL>. Provide no explanations or additional text."),
("human", "Input:"),
("human", "<CORRECT_FUNCTION_CALL> is {function_call}"),
("human", "<COMPLETE_COMMAND> is {command}")
        
    ]

incorrectness_judgement_prompt = \
    [
        ("system","You are AI assistant whose task is to judge."),
        ("human","You have been given a \
                <INCOMPLETE_USER_COMMAND> and a <MODIFIED_INCORRECT_FUNCTION_CALL> that corresponds to it. \
                Your task is to judge whether <MODIFIED_INCORRECT_FUNCTION_CALL> mentions the parameters and parameter values in the <INCOMPLETE_USER_COMMAND> correctly."),
        
        ("human","Guidelines for judgement:"),
        ("human","1. If a parameter and its value is mentioned in <INCOMPLETE_USER_COMMAND>, then the same parameter and value SHOULD BE mentioned in <MODIFIED_INCORRECT_FUNCTION_CALL> correctly."),
        ("human","2.  <MODIFIED_INCORRECT_FUNCTION_CALL> SHOULD NOT HAVE ANY information \
            that is not present in <INCOMPLETE_USER_COMMAND> or CANNOT BE INFERRED from <INCOMPLETE_USER_COMMAND>."),
        ("human","3. If parameter value CAN BE INFERRED from <INCOMPLETE_USER_COMMAND>, it SHOULD BE present in <MODIFIED_INCORRECT_FUNCTION_CALL>."),
        ("human","4. If a parameter is mentioned in <INCOMPLETE_USER_COMMAND> but its value is not mentioned, then the parameter SHOULD NOT be present in <MODIFIED_INCORRECT_FUNCTION_CALL>"),
        ("human","Function name SHOULD NOT play any role in the judgement."),
        ("human","You have to judge only on the basis of parameters and their values between <INCOMPLETE_USER_COMMAND> and <MODIFIED_INCORRECT_FUNCTION_CALL>."),
        ("human","Parameters you should be concerned about, for this judgement are: {parameters}"),
        
        ("human","Output True or False and the reason for your judgement"),
        
        
        ("human","<INCOMPLETE_USER_COMMAND> is {incomplete_user_command}"),
        ("human","<MODIFIED_INCORRECT_FUNCTION_CALL> is {modified_incorrect_function_call}"),
    ]
