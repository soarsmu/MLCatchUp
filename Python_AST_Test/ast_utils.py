import astunparse
import ast
from ast import *
def print_code(node):
    print(astunparse.unparse(node))