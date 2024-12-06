import os
import json
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils import convert_command
from config import ErrorType, functions

class Evaluator:
    def __init__(self, result_file):
        self.result_file = result_file
        self.results = []
        self.eval_result = {}

    def load_results(self):
        """Loads the results from the specified JSON file."""
        with open(self.result_file, "r") as f:
            self.results = json.load(f)

    def save_eval_results(self):
        """Saves the evaluation results to a JSON file."""
        output_file = self.result_file.replace(".json", "_eval_metrics.json")
        with open(output_file, "w") as f:
            json.dump(self.eval_result, f)

    def process_results(self):
        """Processes each output in the results."""
        for output in self.results:
            self.process_output(output)

    def process_output(self, output):
        """Processes a single output."""
        gt = output['assistant']
        pred = output['model_response']
        user_command = output['user']

        gt_fn_call = convert_command(gt)
        pred_fn_call = convert_command(pred)

        if 'error' in gt_fn_call:
            self.eval_result.setdefault('gt_defunctioning_error', []).append(gt)
            return

        if 'error' in pred_fn_call:
            self.eval_result.setdefault('pred_defunctioning_error', []).append(pred)
            return

        sample_res = {'gt': gt, 'pred': pred, 'user_command': user_command}
        gt_fn = gt_fn_call['fn_name']
        self.eval_result.setdefault(gt_fn, {'total': 0, 'correct': 0})
        self.eval_result[gt_fn]['total'] += 1

        if gt_fn_call['fn_name'] == pred_fn_call['fn_name']:
            sample_res['fn_match'] = True
            arg_match = self.compare_properties(gt_fn, gt_fn_call['properties'], pred_fn_call['properties'], sample_res)
            sample_res['arg_match'] = arg_match
            if arg_match:
                self.eval_result[gt_fn]['correct'] += 1
        else:
            if 'possibly_incorrect' in gt_fn_call['fn_name'].lower() and 'possibly_incorrect' in pred_fn_call['fn_name'].lower():
                sample_res['fn_match'] = True
                arg_match = self.compare_properties(gt_fn, gt_fn_call['properties'], pred_fn_call['properties'], sample_res)
                sample_res['arg_match'] = arg_match
                if arg_match:
                    self.eval_result[gt_fn]['correct'] += 1
            else:
                sample_res['fn_match'] = False
                self.append_function_error(sample_res, gt_fn_call['fn_name'], pred_fn_call['fn_name'])

        self.eval_result[gt_fn].setdefault('samples', []).append(sample_res)

    def compare_properties(self, gt_fn, gt_properties, pred_properties, sample_res):
        """Compares the properties of the ground truth and predicted function calls."""
        areEqual = False
        
        if gt_properties == pred_properties:
            return True
        else:
            for key in gt_properties:
                # import pdb; pdb.set_trace()
                if key not in pred_properties:
                    self.append_error_to_sample(sample_res, ErrorType.MISSING_PARAMETER, key, gt_properties[key], None)
                elif gt_properties[key] != pred_properties[key]:
                    fn_details = next(fn for fn in functions if fn['name'] == gt_fn)
                    param_details = fn_details['parameters']['properties'][key]
                    param_type = param_details['type']
                    if 'array' not in param_type:
                        if 'enum' in param_details and pred_properties[key] not in param_details['enum']:
                            self.append_error_to_sample(sample_res, ErrorType.HALLUCINATED_PARAMETER_VALUE, key, gt_properties[key], pred_properties[key])
                        else:
                            self.append_error_to_sample(sample_res, ErrorType.INCORRECT_PARAMETER_VALUE, key, gt_properties[key], pred_properties[key])
                    else:
                        gt_values = gt_properties[key]
                        pred_values = pred_properties[key]
                        if not isinstance(pred_values, list):
                            self.append_error_to_sample(sample_res, ErrorType.INCORRECT_PARAMETER_TYPE_ARRAY, key, gt_values[0], pred_values)
                        else:
                            gt_values.sort()
                            pred_values.sort()
                            # loop through all values of pred_values and check if they are in gt_values
                            for pred_value in pred_values:
                                if pred_value not in gt_values:
                                    if 'enum' in param_details and pred_value not in param_details['enum']:
                                        self.append_error_to_sample(sample_res, ErrorType.HALLUCINATED_ARRAY_ELEMENT, key, gt_values, pred_values)
                                    else:
                                        self.append_error_to_sample(sample_res, ErrorType.INCORRECT_ARRAY_ELEMENT, key, gt_values, pred_values)
                            
                            # loop through all values of gt_values and check if they are in pred_values
                            for gt_value in gt_values:
                                if gt_value not in pred_values:
                                    self.append_error_to_sample(sample_res, ErrorType.MISSING_ARRAY_ELEMENT, key, gt_values, pred_values)        
                            

            for key in pred_properties:
                if key not in gt_properties:
                    self.append_error_to_sample(sample_res, ErrorType.HALLUCINATED_PARAMETER, key, None, pred_properties[key])

            if len(sample_res.get('errors', [])) == 0:
                areEqual = True
            
            return areEqual

    def append_error_to_sample(self, sample_res, error_type, key, gt_value, pred_value):
        """Appends an error to the sample result."""
        sample_res.setdefault('errors', []).append({
            'error_type': error_type.value,
            'key': key,
            'gt_value': gt_value,
            'pred_value': pred_value
        })

    def append_function_error(self, sample_res, gt_fn_name, pred_fn_name):
        """Appends a function-level error to the sample result."""
        sample_res.setdefault('errors', [])
        allowed_pred_fns = [fn['name'] for fn in functions]
        if pred_fn_name not in allowed_pred_fns:
            sample_res['errors'].append({
                'error_type': ErrorType.HALLUCINATED_FUNCTION.value,
                'key': None,
                'gt_value': gt_fn_name,
                'pred_value': pred_fn_name
            })
        else:
            sample_res['errors'].append({
                'error_type': ErrorType.INVALID_FUNCTION.value,
                'key': None,
                'gt_value': gt_fn_name,
                'pred_value': pred_fn_name
            })

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--result_file", type=str, default="./data/sample_predicted_outputs.json")
    args = argparser.parse_args()
    result_file = args.result_file

    evaluator = Evaluator(result_file)
    evaluator.load_results()
    evaluator.process_results()
    evaluator.save_eval_results()

if __name__ == "__main__":
    main()