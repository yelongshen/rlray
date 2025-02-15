import sys
import io
from math_verify import parse, verify
  
print('verify123:', verify(parse("${1,3} \\cup {2,4}$"), parse("${1,2,3,4}$")))
