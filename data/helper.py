import json
import random
import os
from PIL import Image
from itertools import product
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from domain import SYM2PROG, Program, NULL_VALUE
import sys

splits = ['train', 'val', 'test']
symbol_images_dir = 'symbol_images/'

min_num = 0
max_num = 10
num_list = list(map(str, range(min_num, max_num)))
op_list = ['+', '-', '*', '/']
lps = '('
rps = ')'


def render_img(img_paths):
    images = [Image.open(symbol_images_dir + x) for x in img_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('L', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        offset = (i - n_bars / 2) * bar_width + bar_width / 2
        x = np.arange(len(values)) + offset 
        
        ax.bar(x, values, width=bar_width * single_width, color=colors[i % len(colors)], label=name)
        
        
class Iterator:
    def __init__(self, l, shuffle=True):
        if shuffle: 
            random.shuffle(l)
        self.list = l
        self.current = -1
        self.shuffle = shuffle
    
    def next(self):
        self.current += 1
        if self.current == len(self.list):
            self.current = 0
            if self.shuffle:
                random.shuffle(self.list)
        return self.list[self.current]

tok_convert = {'*': 'times', '/': 'div'}
def generate_img_paths(tokens, sym_set):
    img_paths = []
    for tok in tokens:
        if tok in tok_convert:
            tok = tok_convert[tok]
        
        img_name = sym_set[tok].next()
        img_paths.append(os.path.join(tok, img_name))   
        
    return img_paths

class Node:
    def __init__(self, index, symbol, smt, prob=0.):
        self.index = index
        self.symbol = symbol
        self.smt = smt
        self.children = []
        self.sym_prob = prob
        self._res = None
        self._res_computed = False

    def res(self):
        if self._res_computed:
            return self._res

        self._res = self.smt(*self.inputs())
        if isinstance(self._res, int) and self._res > sys.maxsize:
            self._res = None
        self.prob = self.sym_prob + np.log(self.smt.likelihood) + sum([x.prob for x in self.children])
        self._res_computed = True
        return self._res

    def inputs(self):
        return [x.res() for x in self.children if x.res() is not NULL_VALUE]

class AST: # Abstract Syntax Tree
    def __init__(self, pt, semantics):
        self.pt = pt
        self.semantics = semantics

        nodes = [Node(i, s, semantics[s]) for i, s in enumerate(pt.sentence)]

        for node, h in zip(nodes, pt.head):
            if h == -1:
                self.root_node = node
                continue
            nodes[h].children.append(node)
        self.nodes = nodes

        self.root_node.res()
    
    def res(self): return self.root_node.res()
    
    def res_all(self): return [nd._res for nd in self.nodes]
    
op2precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '!': 3}
sym2arity = {'+': 2, '-': 2, '*': 2, '/': 2, '!': 1}
sym2arity.update({n: 0 for n in num_list})

def parse_infix(expr):
    values = []
    operators = []
    
    head = [-1] * len(expr)
    for (i,sym) in enumerate(expr):
        if sym == lps:
            operators.append(i)
        elif sym == rps:
            while expr[operators[-1]] != lps:
                op = operators.pop()
                for _ in range(sym2arity[expr[op]]):
                    head[values.pop()] = op
                values.append(op)
            i_lps = operators[-1]
            i_rps = i
            head[i_lps] = op
            head[i_rps] = op
            operators.pop()
        elif sym2arity[sym] == 0:
            values.append(i)
        else:
            while len(operators) > 0 and expr[operators[-1]] != lps and \
                op2precedence[expr[operators[-1]]] >= op2precedence[sym]:
                op = operators.pop()
                for _ in range(sym2arity[expr[op]]):
                    head[values.pop()] = op
                values.append(op)
            operators.append(i)

    while len(operators) > 0:
        op = operators.pop()
        for _ in range(sym2arity[expr[op]]):
            head[values.pop()] = op
        values.append(op)

    root_op = values.pop()
    head[root_op] = -1
    assert len(values) == 0

    return head

def parse_prefix(expr):
    head = [-1] * len(expr)
    arity = [sym2arity.get(x, 0) for x in expr]
    for i in range(len(expr)):
        if i == 0: 
            head[i] = -1
            continue
        for j in range(i-1, -1, -1):
            if arity[j] > 0:
                break
        head[i] = j
        arity[j] -= 1
        #print(i, head, arity)

    return head

def flatten(expr):
    if len(expr) == 1:
        return expr
    return [y for x in expr for y in flatten(x)]

def prefix2infix(prefix):
    prefix = list(prefix)
    values = []
    while len(prefix) > 0:
        sym = prefix.pop()
        arity = sym2arity[sym]
        if arity == 0:
            values.append([sym])
        else:
            precedence = op2precedence[sym]
            
            left = values.pop()
            right = values.pop() if arity == 2 else []
            """
            add parenthesis when:
            (1) left is a compound expression and its operator's precedence < the current operator
            (2) right is a compound expression and its operator's precedence <= the current operator
            here we assume that the operator is left-associative.
            """
            if len(left) > 1 and op2precedence[left[1]] < precedence:
                left = [lps] + left + [rps]
            if len(right) > 1 and op2precedence[right[1]] <= precedence:
                right = [lps] + right + [rps]
                
            new_value = [left, sym, right]
            values.append(new_value)
    
    infix = ''.join(flatten(values.pop()))
    assert len(values) == 0
    return infix

from collections import namedtuple
Parse = namedtuple('Parse', 'sentence head')
def eval_expr(expr, head):
    ast = AST(Parse(expr, head), SYM2PROG)
    return ast.res(), ast.res_all()
    
def eval_expr_by_eval(expr):
    expr_for_eval = []
    for symbol in expr:
        if symbol == '!':
            expr_for_eval[-1] = 'math.factorial(' + expr_for_eval[-1] + ')'
        elif symbol == '/':
            expr_for_eval.append('//')
        else:
            expr_for_eval.append(symbol)
#     try:
#         res = eval("".join(expr_for_eval))
#     except OverflowError:
#         res = None
    res = eval("".join(expr_for_eval))
    return res



# expr = '6*(5-2)'
# head = parse_infix(expr)
# res = eval_expr(expr, head)
# print(expr, res, head)

# expr = '/+64-31'
# head = parse_prefix(expr)
# res = eval_expr(expr, head)
# print(expr, res, head)

# expr = prefix2infix(expr)
# head = parse_infix(expr)
# res = eval_expr(expr, head)
# print(expr, res, head)

def enumerate_expression(n_op):
    if n_op == 0:
        return [[x] for x in num_list]
    
    expressions = []
    
    arity = 1
    ops = [op for op in op_list if sym2arity[op] == arity]
    inputs = enumerate_expression(n_op-1)
    expressions.extend(product(ops, inputs))
    
    arity = 2
    ops = [op for op in op_list if sym2arity[op] == arity]
    inputs = []
    for i in range(n_op):
        input_1 = enumerate_expression(i)
        input_2 = enumerate_expression(n_op - 1 - i)
        inputs.extend(product(input_1, input_2))
    expressions.extend(product(ops, inputs))
    
    expressions = [flatten(x) for x in expressions]
    return expressions

def sample_one_expr(n_op):
    if n_op == 0:
        return [random.choice(num_list)]
    op = random.choice(op_list)
    if sym2arity[op] == 1:
        return [op] + sample_one_expr(n_op-1)
    else:
        i = random.randint(0, n_op-1)
        j = n_op - 1 - i
        return [op] + sample_one_expr(i) + sample_one_expr(j)

def sample_expression(n_op, n_instances, min_value=0, max_value=float('inf'), res_max_ratio=1.0):
    max_ins_res = int(n_instances * res_max_ratio)
    res2n_ins = {}
    expressions = []
    with tqdm(total=n_instances) as pbar:
        while len(expressions) < n_instances:
            expr = sample_one_expr(n_op)
            if expr in expressions:
                continue
            head = parse_prefix(expr)
            res, res_all = eval_expr(expr, head)
            if not res:
                continue
            if res not in res2n_ins:
                res2n_ins[res] = 0
            if res2n_ins[res] >= max_ins_res:
                continue
            
            max_res = max([x for x in res_all if x is not None])
            if max_res >= min_value and max_res <= max_value:
                expressions.append(expr)
                res2n_ins[res] += 1
                pbar.update(1)
    return expressions

def generate_expression(n_op, n_instances, min_value=0, max_value=float('inf'), res_max_ratio=1.0):
    if n_op <= 2:
        expressions = []
        for expr in enumerate_expression(n_op):
            head = parse_prefix(expr)
            res, res_all = eval_expr(expr, head)
            if res is not None:
                max_res = max([x for x in res_all if x is not None])
                if max_res >= min_value and max_res <= max_value:
                    expressions.append(expr)
    else:
        assert n_instances is not None
        expressions = sample_expression(n_op, n_instances, min_value, max_value, res_max_ratio)
    random.shuffle(expressions)
    expressions = expressions[:n_instances]
    
    temp = []
    for expr in expressions:
        expr = prefix2infix(expr)
        head = parse_infix(expr)
        res, res_all = eval_expr(expr, head)
        if res is not None:
            temp.append((expr, head, res, res_all))
    
    return temp