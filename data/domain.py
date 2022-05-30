DIGITS= [str(i) for i in range(0, 10)]
OPERATORS = list('+-*/')
PARENTHESES = list('()')
START = '<START>'
END = '<END>'
NULL = '<NULL>'
SYMBOLS = DIGITS + OPERATORS + PARENTHESES
# SYM2ID = {v:i for i, v in enumerate(SYMBOLS)}
# ID2SYM = {i:v for i, v in enumerate(SYMBOLS)}
SYM2ID = lambda x: SYMBOLS.index(x) if x in SYMBOLS else None
ID2SYM = lambda x: SYMBOLS[x]

FEWSHOT_OPERATORS = list('abcd')
FEWSHOT_NUMBERS = list('xy')

NULL_VALUE = -1

import math
from inspect import signature
class Program():
    def __init__(self, fn=None):
        self.fn = fn
        self.arity = len(signature(fn).parameters) if fn is not None else 0
        self.likelihood = 1.0
        self.cache = {} # used for fast computation

    def __call__(self, *inputs):
        if len(inputs) != self.arity or None in inputs:
            return None
        res = self.fn(*inputs)
        self.cache[inputs] = res
        return res

    def evaluate(self, examples, store_y=True): 
        ys = []
        for exp in examples:
            try:
                y = self(*exp)
            except (TypeError, RecursionError) as e:
                y = None
            ys.append(y)
        return ys

    def solve(self, i, inputs, output_list):
        if len(inputs) != self.arity:
            return []
        
        def equal(a, b, pos):
            for j in range(len(a)):
                if j == pos:
                    continue
                if a[j] != b[j]:
                    return False
            return True

        candidates = []
        for xs, y in self.cache.items():
            if y in output_list and equal(xs, inputs, i):
                candidates.append(xs[i])
        return candidates

functions = [
    lambda: 0, lambda: 1, lambda: 2, lambda: 3, lambda: 4, lambda: 5, lambda: 6, lambda: 7, lambda: 8, lambda: 9,
    lambda x,y: x+y, lambda x,y: max(0, x-y), lambda x,y: x*y, lambda x,y: math.ceil(x/y) if y != 0 else None, 
    lambda: NULL_VALUE, lambda: NULL_VALUE,
]

PROGRAMS = [Program(f) for f in functions] 
SYM2PROG= {s:p for s, p in zip(SYMBOLS, PROGRAMS)}