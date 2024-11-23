1. Generate data --> python generate_training_data.py --no_prompt_insertion (if you want data to train without inserting function in prompt -- the way we want) or
  python generate_training_data.py --prompt_insertion (if you want data to train with injecting function (1 function right now, can be tweaked easily for more) --> way in which function
calling is typically trained.)
   
2. Finetune the model --> python function_call_finetune.py
3. Use it for prediction --> python function_call_predict.py

It is imperfect in a lot of ways. PRs are welcomed. I ll probably switch my focus to something else. 

