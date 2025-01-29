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
  
response = ['\nTo find the range of the function $p(x)$, we need to consider the two cases given in the definition of the function.\n\n**Case 1: $\\lfloor x \\rfloor$ is prime**\n\nIn this case, the function is defined as $p(x) = x + 1$. Since $2 \\le x \\le 10$, the prime numbers in this range are $2, 3, 5, 7$. For each of these primes, the function simply adds $1$ to the value of $x$. Therefore, the range for this case is:\n\n$$\\{2+1, 3+1, 5+1, 7+1\\} = \\{3, 4, 6, 8\\}$$\n\n**Case 2: $\\lfloor x \\rfloor$ is not prime**\n\nIn this case, the function is defined recursively as $p(x) = p(y) + (x + 1 - \\lfloor x \\rfloor)$, where $y$ is the greatest prime factor of $\\lfloor x \\rfloor$. We need to consider the non-prime numbers in the range $2 \\le x \\le 10$, which are $4, 6, 8, 9, 10$.\n\n- For $x = 4$, the greatest prime factor is $2$, so $p(4) = p(2) + (4 + 1 - 4) = p(2) + 1$. Since $p(2) = 2 + 1 = 3$, we have $p(4) = 4$.\n- For $x = 6$, the greatest prime factor is $3$, so $p(6) = p(3) + (6 + 1 - 6) = p(3) + 1$. Since $p(3) = 3 + 1 = 4$, we have $p(6) = 5$.\n- For $x = 8$, the greatest prime factor is $2$, so $p(8) = p(2) + (8 + 1 - 8) = p(2) + 1$. Since $p(2) = 3$, we have $p(8) = 4$.\n- For $x = 9$, the greatest prime factor is $3$, so $p(9) = p(3) + (9 + 1 - 9) = p(3) + 1$. Since $p(3) = 4$, we have $p(9) = 5$.\n- For $x = 10$, the greatest prime factor is $5$, so $p(10) = p(5) + (10 + 1 - 10) = p(5) + 1$. Since $p(5) = 5 + 1 = 6$, we have $p(10) = 7$.\n\nCombining the results from both cases, the range of $p(x)$ is:\n\n$$\\{3, 4, 5, 6, 7\\}$$\n\nIn interval notation, since the range consists of distinct integers, we can express it as a union of intervals:\n\n$$\\boxed{[3, 3] \\cup [4, 4] \\cup [5, 5] \\cup [6, 6] \\cup [7, 7]}$$\n\nHowever, since each interval contains only a single point, it is more conventional to simply list the points without interval notation:\n\n$$\\boxed{[3, 4, 5, 6, 7]}$$\n\nThis is the range of the function $p(x)$ for $2 \\le x \\le 10$.\n\nThe answer is: [3, 4, 5, 6, 7].<|end_of_solution|> \n\n\n']

pattern = r"The answer is: \\boxed\{(.*?)\}"

answers = ['[3,7] \\cup [8,9)']

for query, answer in zip(response, answers):
    n_query, p_answer, box_match = preprocess_orm800k_box_responsev1(query, answer)
    print('raw response', query)
    print('new response', n_query)
    print('extract answer', p_anwer)
    print('gt answer', answer)
    print('box match', box_match)
  
