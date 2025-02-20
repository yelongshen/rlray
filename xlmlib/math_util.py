import os
import math
import io
import re
import sys
import subprocess
import pickle

from latex2sympy.latex2sympy2 import latex2sympy
from math_evaluation import is_equiv
#from .math_verify_util import math_verify

def math_verify(gold, answer):
    escaped_answer = answer.replace("\n","\\n").replace("\r","\\r").replace("\\", "\\\\").replace('"', '\\"')
    escaped_gold = gold.replace("\n","\\n").replace("\r","\\r").replace("\\", "\\\\").replace('"', '\\"')
    command = ['python3.12', '-c', f'''from math_verify import parse, verify; import pickle; import sys; pickle.dump(verify(parse("{escaped_answer}"), parse("{escaped_gold}")), sys.stdout.buffer)''']
    o = False
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        o = pickle.load(proc.stdout)
    return o
    
def is_numeric(s):
    """Check if a string is a valid numeric value (integer or float)."""
    try:
        float(s)  # Try converting to float
        return True
    except ValueError:
        return False

#Here is a Python program that checks if two input strings are numeric strings and then compares their numerical values:
def compare_math_answers(gt, pred):
    """Compare two numeric strings and print the result."""
    if not is_numeric(gt) or not is_numeric(pred):
        #print("Both inputs must be numeric strings!")
        if gt.strip() == pred.strip():
            return True
        else:
            return False

    # Convert both to floats for comparison
    num1, num2 = float(gt), float(pred)

    if num1 == num2:
        return True
    else:
        return False
      
def process_math_prompt(original_question, prompt_type = "v8"):
    
    candidate_prompt_1 = "Think step by step and provide the final answer at the end in this format: 'The final answer is: <your_answer>'\n"

    candidate_prompt_2 = "Think step by step and conclude with the final answer in this format: 'The final answer is: <your_answer>' Ensure <your_answer> is the simplest form. \n\n"

    candidate_prompt_3 = "Think step by step and provide the answer at the end in this format: 'The answer is: <your_answer>'\n"

    candidate_prompt_4 = "Please think step by step first and then conclude the answer at the end in this format: 'The answer is: <your_answer>'\n"

    candidate_prompt_5 = "First, think step by step to carefully analyze the problem. Then, conclude with the final answer in this format: 'The answer is: <your_answer>'\n"

    candidate_prompt_6 = "First, think step by step to carefully analyze the problem. Then, conclude with the answer in this format: 'The answer is: <your_answer>'\n\n"

    candidate_prompt_7 = "First, think step by step. Then, provide a concise, one-line answer in this format: 'The answer is: <your_answer>'. \n\n"

    candidate_prompt_8 = "First, think step by step to carefully analyze the problem. Then, conclude a concise, one-line answer at the end in this format: 'The answer is: <your_answer>' \n\n"

    candidate_prompt_9 = "You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}:\n"
    #pattern = r'The answer is:\s*(.+)'

    postfix_instruct = ''
    
    if prompt_type == 'v8':
        prefix_instruct = candidate_prompt_8 #if prompt_type == 'v8' else candidate_prompt_4    
    elif prompt_type == 'v9':
        prefix_instruct = candidate_prompt_9
        
    prompt = prefix_instruct + original_question + postfix_instruct

    return prompt

def process_math_answer(response, answers, tokenizer, prompt_type = "v8"):
    try:
        split_response = response.strip().split('\n')[-1]
        if math_verify(answers[0], split_response):
            return response, answers[0], 1.0
    except:
        print('error response:', response)
        
    pattern = r'The answer is:\s*(.+)'

    box_match = 0.0
    extracted_answer = 'none'
        
    match = re.search(pattern, response, re.MULTILINE)
        
    if match:
        extracted_answer = match.group(1) #or match.group(2) or match.group(3) or match.group(4)
        # clean up special tokens.
        answer_tokens = tokenizer([extracted_answer], add_special_tokens=False, max_length=1024, truncation=True)
        extracted_answer = tokenizer.decode(answer_tokens['input_ids'][0], skip_special_tokens=True)
        #     print('verify', math_verify("${1,3} \\cup {2,4}$", "${1,2,3,4}$")) 
        for ans in answers:
            is_match = compare_math_answers(ans, extracted_answer) or is_equiv(ans, extracted_answer) or math_verify(extracted_answer, ans)
            if is_match:
                box_match = 1.0
                break
                
        pos = match.end() 
        response = response[:pos]

    return response, extracted_answer, box_match

