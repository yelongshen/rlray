import os
import math
import io
import re
import sys
import subprocess
import pickle
import concurrent.futures
import time
import multiprocessing

from transformers.utils import logging

logger = logging.get_logger(__name__)

try:
    from math_evaluation import is_equiv
except ImportError as error:
    logger.warning(
        f"`math_evaluation` package not found, consider installing for better performance: {error}."
    )

def math_verify(gold, answer):
    escaped_answer = answer.replace("\n","\\n").replace("\r","\\r").replace("\\", "\\\\").replace('"', '\\"')
    escaped_gold = gold.replace("\n","\\n").replace("\r","\\r").replace("\\", "\\\\").replace('"', '\\"')
    command = ['python3.12', '-c', f'''from math_verify import parse, verify; import pickle; import sys; pickle.dump(verify(parse("{escaped_answer}"), parse("{escaped_gold}")), sys.stdout.buffer)''']
    o = False

    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
            try:
                o, _ = proc.communicate(timeout=5)  # Wait for output with timeout
                return pickle.loads(o)
            except subprocess.TimeoutExpired:
                proc.kill()  # Kill process if it times out
                print("Process timed out!")
            except pickle.UnpicklingError:
                print("Failed to unpickle the output.")
            except Exception as e:
                print(f"An error occurred during subprocess communication: {e}")
    except Exception as e:
        print(f"An error occurred while executing the subprocess: {e}")

    return False
        
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
    candidate_prompt_10 = "You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:\n"

    candidate_prompt_11 = r"You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}:\n"

    candidate_prompt_13 = r"You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}:\n USER: "

    candidate_prompt_14 = r"You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}:\n <|user|>: "

    candidate_prompt_16 = "You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}:\n<|user|>"

    candidate_prompt_17 = "You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}:\n"

    postfix_instruct = ''
    
    if prompt_type == 'v8':
        prefix_instruct = candidate_prompt_8 #if prompt_type == 'v8' else candidate_prompt_4    
    elif prompt_type == 'v9':
        prefix_instruct = candidate_prompt_9
    elif prompt_type == 'v10':
        prefix_instruct = candidate_prompt_10
    elif prompt_type == 'v11':
        prefix_instruct = candidate_prompt_11
    elif prompt_type == 'v12':
        prefix_instruct = candidate_prompt_11
        postfix_instruct = '\n'
    elif prompt_type == 'v13':
        prefix_instruct = candidate_prompt_13 
        postfix_instruct = '\n ASSISTANT: '
    elif prompt_type == 'v14':
        prefix_instruct = candidate_prompt_14 
        postfix_instruct = '\n <|assistant|>: '
    elif prompt_type == 'v15':
        prefix_instruct = candidate_prompt_14
        postfix_instruct = '\n <|end|> <|assistant|>: '
    elif prompt_type == 'v16':
        prefix_instruct = candidate_prompt_16
        postfix_instruct = '<|end|><|assistant|>'
    elif prompt_type == 'v17':
        prefix_instruct = candidate_prompt_17
        postfix_instruct = '<|end|>'
    prompt = prefix_instruct + original_question + postfix_instruct

    return prompt

#def call_with_timeout_v2(func, *args, timeout=5):
#    with concurrent.futures.ThreadPoolExecutor() as executor:
#        future = executor.submit(func, *args)
#        try:
#            return future.result(timeout=timeout)  # This will raise TimeoutError if the function takes longer than timeout
#        except concurrent.futures.TimeoutError:
#            print("Function call timed out")
#            return False

def safe_math_answer_timeout(response, answers, tokenizer, prompt_type = "v8", alg = ['math_verify', 'is_equiv', 'text', 'lastline_math_verify', 'full_math_verify'], timeout=30):
    r = call_with_timeout(process_math_answer, response, answers, tokenizer, prompt_type, alg, timeout=timeout)
    if r:
        return r
    else:
        return response, "none", 0.0

def call_with_timeout(func, *args, timeout=2):
    """Runs a function with a timeout. If it exceeds, terminates the process."""
    with multiprocessing.Pool(processes=1) as pool:
        result = pool.apply_async(func, args)
        try:
            return result.get(timeout=timeout)  # Get result with timeout
        except multiprocessing.TimeoutError:
            print("Function call timed out")
            return False  # Or any fallback value

#def call_with_timeout(func, *args, timeout=2):
#    """Runs a function with a hard timeout using multiprocessing."""
#    p = multiprocessing.Process(target=func, args=args)
#    p.start()
#    p.join(timeout)  # Wait for timeout seconds
#    if p.is_alive():
#        print("Function call exceeded timeout. Terminating...")
#        p.terminate()  # Forcefully terminate
#        p.join()
#        return False  # Or any fallback value
#    return True
            
def process_math_answer(response, answers, tokenizer, prompt_type = "v8", alg = ['math_verify', 'is_equiv', 'text', 'lastline_math_verify', 'full_math_verify']):
    if prompt_type == 'v8':
        pattern_prefix = 'The answer is:'
        pattern = r'The answer is: \s*(.+)'
    elif prompt_type == 'v9' or prompt_type == 'v10':
        pattern_prefix = 'answer is:'
        pattern = r'answer is: \\boxed\{(.*?)\}'
    elif prompt_type == 'v11' or prompt_type == 'v12' or prompt_type == 'v13' or prompt_type == 'v14' or prompt_type == 'v15' or prompt_type == 'v16' or prompt_type == 'v17':
        pattern_prefix = ''
        #pattern = r'\\boxed{([^}]*)}'
        #pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
        pattern = r'\\boxed\{((?:[^{}]+|{(?:[^{}]+|{[^{}]*})*})*)\}'
    box_match = 0.0
    extracted_answer = 'none'
    ans = answers[0]
    matches = list(re.finditer(pattern, response, re.MULTILINE)) 
    
    if matches:
        answer_start, answer_end = matches[-1].span() 
        if prompt_type == 'v8':
            extracted_answer = response[answer_start + len(pattern_prefix) : answer_end]
            answer_tokens = tokenizer([extracted_answer], add_special_tokens=False, max_length=1024, truncation=True)
            extracted_answer = tokenizer.decode(answer_tokens['input_ids'][0], skip_special_tokens=True)
        else:
            extracted_answer = matches[-1].group(1)
        #for ans in answers:
        is_match = compare_math_answers(ans, extracted_answer) 
        if not is_match and 'is_equiv' in alg:
            is_match = is_match or is_equiv(ans, extracted_answer) #, timeout=5) # is_equiv(ans, extracted_answer) is_equiv(ans, extracted_answer) # 
        
        if not is_match and 'math_verify' in alg: #  fast_mode == 0 or fast_mode == 1: 
            is_match = is_match or math_verify(ans, extracted_answer)
        
        if not is_match and 'text' in alg:
            if ans.startswith('\\text{'):
                text_match = re.search(r'\\text\{(.*?)\}', ans)
                ans = text_match.group(1)
                is_match = is_match or is_equiv(ans, extracted_answer) #, timeout=5)  # is_equiv(new_ans, extracted_answer) #  #is_equiv(new_ans, extracted_answer)

        if not is_match and ',' in ans:
            multiple_ans = ans.split(',')
            ans_number = len(multiple_ans)
                
            if len(matches) >= ans_number:
                succ = 0
                for t_a in range(0, ans_number):
                    sub_match = 0
                    for m in range(0, ans_number):
                        if is_equiv(multiple_ans[t_a], matches[-1-m].group(1)): #, timeout=1):
                            sub_match = 1    
                            break
                    if sub_match == 0:
                        break
                    succ = succ + sub_match
                if succ >= ans_number:
                    is_match = True
        if is_match:
            box_match = 1.0
        #pos = matches.end() 
        response = response[:answer_end]
        #return response, extracted_answer, box_match
    if box_match < 0.5 and ('lastline_math_verify' in alg or 'full_math_verify' in alg):
        try:
            split_response = response if 'full_math_verify' in alg else response.strip().split('\n')[-1]
            if math_verify(ans, split_response):
                return response, ans, 1.0
        except:
            print('error response:', response)

    return response, extracted_answer, box_match
