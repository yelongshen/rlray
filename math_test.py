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
    $$\dfrac{28}{12} = \dfrac{28 \div 4}{12 \div 4} = \dfrac{7}{3}.$$  The answer is: \\frac{7}{3}.

    Tha answer is \\frac{100}{101}.

    The answer is \\frac{1}{10}.
    '''

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
    
    The answer in base 10 is $\boxed{2938}$.  The answer is: 2938.
    
    The answer is 1290.'''

response_3 = '''To find the range of $p(x)$, we need to consider the two cases given in the definition of $p(x)$:

    1. When $\lfloor x \rfloor$ is prime:
    In this case, $p(x) = x + 1$. Since $2 \le x \le 10$, the possible prime values for $\lfloor x \rfloor$ are 2, 3, 5, and 7. For each of these values, $p(x)$ will be equal to $x + 1$.
    
    2. When $\lfloor x \rfloor$ is not prime:
    In this case, $p(x) = p(y) + (x + 1 - \lfloor x \rfloor)$, where $y$ is the greatest prime factor of $\lfloor x \rfloor$. We need to find the possible values of $y$ and calculate the corresponding values of $p(x)$.
    
    For $\lfloor x \rfloor = 4$, the greatest prime factor is 2, so $p(x) = p(2) + (x + 1 - 4) = 2 + (x - 3)$.
    
    For $\lfloor x \rfloor = 6$, the greatest prime factor is 3, so $p(x) = p(3) + (x + 1 - 6) = 3 + (x - 5)$.
    
    For $\lfloor x \rfloor = 8$, the greatest prime factor is 2, so $p(x) = p(2) + (x + 1 - 8) = 2 + (x - 7)$.
    
    For $\lfloor x \rfloor = 9$, the greatest prime factor is 3, so $p(x) = p(3) + (x + 1 - 9) = 3 + (x - 8)$.
    
    Now, we need to find the minimum and maximum values of $p(x)$ for each case:
    
    1. For $p(x) = x + 1$, the minimum value is $2 + 1 = 3$ and the maximum value is $10 + 1 = 11$.
    
    2. For $p(x) = 2 + (x - 3)$, the minimum value is $2 + (2 - 3) = 1$ and the maximum value is $2 + (10 - 3) = 9$.
    
    3. For $p(x) = 3 + (x - 5)$, the minimum value is $3 + (2 - 5) = -2$ and the maximum value is $3 + (10 - 5) = 8$.
    
    4. For $p(x) = 2 + (x - 7)$, the minimum value is $2 + (2 - 7) = -3$ and the maximum value is $2 + (10 - 7) = 5$.
    
    5. For $p(x) = 3 + (x - 8)$, the minimum value is $3 + (2 - 8) = -3$ and the maximum value is $3 + (10 - 8) = 5$.
    
    Combining all the minimum and maximum values, we have:
    
    Minimum value: -3
    Maximum value: 11
    
    Therefore, the range of $p(x)$ in interval notation is $[-3, 11]$.
    
    The answer is: [-3, 11].
    
    The answer is: [-3, 110].'''

response_4 = '''$P(x) = -\sqrt[3]{2}x^4 + (4\sqrt[3]{2} - 2)x^3 - 3x^2 + 3\sqrt[3]{4}x - 2\sqrt[3]{8}$

Now, we can simplify the polynomial by multiplying each term by $\sqrt[3]{2}$:

$P(x) = -2x^4 + 8x^3 - 6\sqrt[3]{2}x^2 + 6\sqrt[3]{4}x - 4\sqrt[3]{8}$

Since we want a monic polynomial with integer coefficients, we can multiply the entire polynomial by $\sqrt[3]{2}$:

$P(x) = -2\sqrt[3]{2}x^4 + 8\sqrt[3]{2}x^3 - 6x^2 + 6\sqrt[3]{4}x - 4\sqrt[3]{8}$

Finally, we can simplify the polynomial:

$P(x) = -2x^4 + 8x^3 - 6x^2 + 6\sqrt[3]{4}x - 4\sqrt[3]{8}$

The answer is: $P(x) = -2x^4 + 8x^3 - 6x^2 + 6\sqrt[3]{4}x - 4\sqrt[3]{8}$.'''

response_5 = '''$P(x) = -\sqrt[3]{2}x^4 + (4\sqrt[3]{2} - 2)x^3 - 3x^2 + 3\sqrt[3]{4}x - 2\sqrt[3]{8}$

Now, we can simplify the polynomial by multiplying each term by $\sqrt[3]{2}$:

$P(x) = -2x^4 + 8x^3 - 6\sqrt[3]{2}x^2 + 6\sqrt[3]{4}x - 4\sqrt[3]{8}$

Since we want a monic polynomial with integer coefficients, we can multiply the entire polynomial by $\sqrt[3]{2}$:

$P(x) = -2\sqrt[3]{2}x^4 + 8\sqrt[3]{2}x^3 - 6x^2 + 6\sqrt[3]{4}x - 4\sqrt[3]{8}$

Finally, we can simplify the polynomial:

$P(x) = -2x^4 + 8x^3 - 6x^2 + 6\sqrt[3]{4}x - 4\sqrt[3]{8}$

The answer is: $P(x) = -2x^4 + 8x^3 - 6x^2 + 6\sqrt[3]{4}x - 4\sqrt[3]{8}$

The answer is: kjaiadsgdf.'''

response_6 = '''$P(x) = -\sqrt[3]{2}x^4 + (4\sqrt[3]{2} - 2)x^3 - 3x^2 + 3\sqrt[3]{4}x - 2\sqrt[3]{8}$

Now, we can simplify the polynomial by multiplying each term by $\sqrt[3]{2}$:

$P(x) = -2x^4 + 8x^3 - 6\sqrt[3]{2}x^2 + 6\sqrt[3]{4}x - 4\sqrt[3]{8}$

Since we want a monic polynomial with integer coefficients, we can multiply the entire polynomial by $\sqrt[3]{2}$:

$P(x) = -2\sqrt[3]{2}x^4 + 8\sqrt[3]{2}x^3 - 6x^2 + 6\sqrt[3]{4}x - 4\sqrt[3]{8}$

Finally, we can simplify the polynomial:

$P(x) = -2x^4 + 8x^3 - 6x^2 + 6\sqrt[3]{4}x - 4\sqrt[3]{8}$

The answer is: 3.451

The answer is: kjaiadsgdf.'''



response = [response_1, response_2, response_3, response_4, response_5]

#pattern = r"The answer is: \[(.*?)\]"
#pattern = r'The answer is: (?:\[(.*?)\]|(\d+)|"(.*?)")'

#pattern = r'The answer is:\s*(?:\[(.*?)\]|(\d+)|"(.*?)"|\\frac{(\d+)}{(\d+)})'

#pattern = r'The answer is:\s*(?:\[(.*?)\]|(\d+)|"(.*?)"|(\$?\\frac{\d+}{\d+}\$?))'
#pattern = r'The answer is:\s*(.*?)\s*\.?'
pattern = r'The answer is:\s*(\S+)\\n'
#pattern = r'The answer is:\s*(?:\[(.*?)\]|(\d+)|"(.*?)"|(\$?\\frac{\d+}{\d+}\$?)|\[-?\d+,\s*-?\d+\])'
#pattern = r'The answer is:\s*()'
#pattern = r'The answer is:\s*(\[-?\d+,\s*-?\d+\])'

for r in response:
    match = re.search(pattern, r, re.DOTALL)
    if match:
        extracted_answer = match.group(1) # or match.group(2) or match.group(3) or match.group(4) or match.group(5)
        print("Extracted Answer:", extracted_answer)

        pos = match.end() 
        print(r[:pos])
    else:
        print("No match found.")
    
#print('\n\n\nraw response:\n', query)
#print('\n\n\nnew response:\n', n_query)
#print('\n\n\nextract answer:\n', p_answer)
#print('\n\n\ngt answer:\n', answer)
#print('\n\n\nbox match:\n', box_match)
  
