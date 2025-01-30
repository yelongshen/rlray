import os
import math
import io

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
        if gt.strip() == s2.strip():
            return True
        else:
            return False

    # Convert both to floats for comparison
    num1, num2 = float(s1), float(s2)

    if num1 == num2:
        return True
    else:
        return False
      
