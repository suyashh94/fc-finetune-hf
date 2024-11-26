import os 
import json 
import numpy as np
import argparse
from utils import convert_command
from config import ErrorType, functions

argparser = argparse.ArgumentParser()
argparser.add_argument("--result_file", type=str, default="./data/car_finetuning_gpt_False_output.json")

args = argparser.parse_args()
result_file = args.result_file

with open(result_file, "r") as f:
    results = json.load(f)

eval_result = {}

for output in results:
    
    # import pdb; pdb.set_trace()
    
    gt = output['assistant']
    pred = output['model_response']
    
    gt_fn_call = convert_command(gt)
    pred_fn_call = convert_command(pred)
    
    if 'error' in gt_fn_call:
        temp = eval_result.get('gt_defunctioning_error', [])
        temp.append(gt)
        eval_result['gt_defunctioning_error'] = temp
        continue
    
    if 'error' in pred_fn_call:
        temp = eval_result.get('pred_defunctioning_error', [])
        temp.append(pred)
        eval_result['pred_defunctioning_error'] = temp
        continue
    
    sample_res = {
        'gt':gt, 
        'pred':pred,
    }
    
    gt_fn = gt_fn_call['fn_name']
    eval_result[gt_fn] = eval_result.get(gt_fn, {'total': 0, 'correct': 0})
    eval_result[gt_fn]['total'] += 1
    
    if gt_fn_call['fn_name'] == pred_fn_call['fn_name']:
        sample_res['fn_match'] = True
    else:
        sample_res['fn_match'] = False
        sample_res['errors'] = sample_res.get('errors', [])
        
        allowed_pred_fns = list(map(lambda x: x['name'], functions))
        
        if pred_fn_call['fn_name'] not in allowed_pred_fns:
            sample_res['errors'].append({
                'error_type': ErrorType.HALLUCINATED_FUNCTION.value,
                'key': None,
                'gt_value': gt_fn_call['fn_name'],
                'pred_value': pred_fn_call['fn_name']
            })
        else:
            sample_res['errors'].append({
                'error_type': ErrorType.INVALID_FUNCTION.value,
                'key': None,
                'gt_value': gt_fn_call['fn_name'],
                'pred_value': pred_fn_call['fn_name']
            })
    
    if sample_res['fn_match'] == True:    
    
        if gt_fn_call['properties'] == pred_fn_call['properties']:
            sample_res['arg_match'] = True
            eval_result[gt_fn]['correct'] += 1
        else:
            for key in gt_fn_call['properties']:
                sample_res['arg_match'] = False
                if key not in pred_fn_call['properties']:
                    sample_res['errors'] = sample_res.get('errors', [])
                    sample_res['errors'].append({
                        'error_type': ErrorType.MISSING_PARAMETER.value,
                        'key': key,
                        'gt_value': gt_fn_call['properties'][key],
                        'pred_value': None
                    })
                elif gt_fn_call['properties'][key] != pred_fn_call['properties'][key]:
                    sample_res['errors'] = sample_res.get('errors', [])
                    
                    
                    fn_details = list(filter(lambda x: x['name'] == gt_fn, functions))[0]
                    if 'enum' in fn_details['parameters']['properties'][key] and pred_fn_call['properties'][key] not in fn_details['parameters']['properties'][key]['enum']:
                        sample_res['errors'].append({
                            'error_type': ErrorType.HALLUCINATED_PARAMETER_VALUE.value,
                            'key': key,
                            'gt_value': gt_fn_call['properties'][key],
                            'pred_value': pred_fn_call['properties'][key]
                        })
                        
                    else:
                        sample_res['errors'].append({
                            'error_type': ErrorType.INCORRECT_PARAMETER_VALUE.value,
                            'key': key,
                            'gt_value': gt_fn_call['properties'][key],
                            'pred_value': pred_fn_call['properties'][key]
                        })
            
            for key in pred_fn_call['properties']:
                if key not in gt_fn_call['properties']:
                    sample_res['errors'] = sample_res.get('errors', [])
                    sample_res['errors'].append({
                        'error_type': ErrorType.HALLUCINATED_PARAMETER.value,
                        'key': key,
                        'gt_value': None,
                        'pred_value': pred_fn_call['properties'][key]
                    })
            
    try:    
        curr_samples = eval_result[gt_fn].get('samples', [])
        curr_samples.append(sample_res)
        eval_result[gt_fn]['samples'] = curr_samples
    except:
        import pdb; pdb.set_trace()

# import pdb; pdb.set_trace()

with open(result_file.replace(".json", "_eval_metrics.json"), "w") as f:
    json.dump(eval_result, f)

# import pdb; pdb.set_trace()
    

    

