import re
import io
import sys


def preprocess_orm800k_box_responsev1(sequence, answer):
    temp_query = ""
    temp_response = ""
    temp_query = sequence.split("ASSISTANT:\n")[0]
    #temp_query = temp_query.split("# Question\n\n")[-1]
    temp_response = sequence.split("ASSISTANT:\n")[-1]
    #print("temp_response", temp_response)
    # 使用正则表达式匹配
    # pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    # # 正则表达式，处理换行与特殊字符
    # pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    #pattern = r"The final answer is: \\\\boxed\{(.*?)\}"
    pattern = r"The answer is: \\boxed\{(.*?)\}"
    # 使用 re.DOTALL 确保能匹配跨行文本
    match = re.search(pattern, temp_response, re.DOTALL)
    #match = re.search(pattern, temp_response)
    if match:
        temp_answer = match.group(1)
    else:
        temp_answer = "none"
    #temp_answer = temp_response.split("\n\n# Answer\n\n")[-1]
    temp_response = sequence.split("<|reserved_special_token_0|>The answer is:")[0]
    #response_list = temp_response.split("<|reserved_special_token_0|>")
    processed_solution = temp_response + "\n\n# Answer\n\n" + temp_answer + "<|reserved_special_token_0|>"
    # for i in range(len(response_list)-1):
    #     processed_solution = processed_solution + "Step " + str(i+1) + ": " + response_list[i] + " <|reserved_special_token_0|>\n"
    # processed_solution = processed_solution + response_list[-1] + " The answer is: " + temp_answer + " <|reserved_special_token_0|>\n"
    processed_solution = re.sub(r"<\|end_of_text\|>", "", processed_solution)
    if temp_answer == answer:
        box_match = 1.0 # 0.5
    else:
        box_match = 0.0 # -0.5
    return processed_solution, temp_answer, box_match

response_1 = '''To add these fractions, we need a common denominator. The least common multiple of 12 and 4 is 12. So, we convert $\dfrac{5}{4}$ to have a denominator of 12:
    $$\dfrac{13}{12} + \dfrac{5 \times 3}{4 \times 3} = \dfrac{13}{12} + \dfrac{15}{12}.$$
    
    Now we can add the fractions:
    $$\dfrac{13}{12} + \dfrac{15}{12} = \dfrac{13 + 15}{12} = \dfrac{28}{12}.$$
    
    Finally, we simplify the fraction by dividing both the numerator and denominator by their greatest common divisor, which is 4:
    $$\dfrac{28}{12} = \dfrac{28 \div 4}{12 \div 4} = \dfrac{7}{3}.$$ 
    
    The answer is: \frac{7}{3}.'''

response_2 = '''To solve this problem, we need to convert each number to base 10, perform the arithmetic operations, and then convert the result back to base 10 if necessary.
    
    First, let's convert each number to base 10:
    
    $1357_{9}$ in base 10:
    $1 \times 9^3 + 3 \times 9^2 + 5 \times 9^1 + 7 \times 9^0 = 729 + 243 + 45 + 7 = 1024$
    
    $100_{4}$ in base 10:
    $1 \times 4^2 + 0 \times 4^1 + 0 \times 4^0 = 16 + 0 + 0 = 16$
    
    $2460_{8}$ in base 10:
    $2 \times 8^3 + 4 \times 8^2 + 6 \times 8^1 + 0 \times 8^0 = 1024 + 256 + 48 + 0 = 1328$
    
    $5678_{9}$ in base 10:
    $5 \times 9^3 + 6 \times 9^2 + 7 \times 9^1 + 8 \times 9^0 = 3645 + 486 + 63 + 8 = 4202$
    
    Now, let's perform the arithmetic operations:
    
    $\frac{1024}{16} - 1328 + 4202 = 64 - 1328 + 4202 = -1264 + 4202 = 2938$
    
    The answer in base 10 is $\boxed{2938}$.
    
    The answer is: 2938.'''

response = [response_1, response_2]

pattern = r"The answer is: \[(.*?)\]"

for r in response:
    # Search for the pattern
    match = re.search(pattern, r, re.DOTALL)
    # Extract and print the answer
    if match:
        extracted_answer = match.group(1)  # Extracts "3, 4, 5, 6, 7"
        print("Extracted Answer:", extracted_answer)
    else:
        print("No match found.")
    
#print('\n\n\nraw response:\n', query)
#print('\n\n\nnew response:\n', n_query)
#print('\n\n\nextract answer:\n', p_answer)
#print('\n\n\ngt answer:\n', answer)
#print('\n\n\nbox match:\n', box_match)
  
