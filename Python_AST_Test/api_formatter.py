import ast
from astunparse import unparse
from ast import Assign
from Python_AST_Test.ast_utils import *

def separate_api_parameter(node):
    # Loop through the given node from assignment to get all the API Call and keyword
    print()
    print()
    print("FULL ORIGINAL NODE: " + unparse(node))
    api_name = getName(node)
    api_keywords = []
    api_keywords.extend(getKeywordArguments(node))
    scope_list = recurseScope(node)
    for scope in scope_list:
        if scope is not None:
            print("Processed scope1: " + unparse(scope))
            print("Processed scope2: " + ast.dump(scope))
            api_keywords.extend(getKeywordArguments(scope))
            api_name = getName(scope) + "." + api_name

    print("Full API Name: " + api_name)
    print("FULL API KEYWORD: " + api_keywords.__str__())
    separated_signature = api_name + "#"
    last_element = api_keywords[:-1]
    for key in api_keywords:
        separated_signature += key
        if key != last_element:
            separated_signature += ", "
    print("Separated Signature: " + separated_signature)
    print()
    print()

    api_keywords = getKeywordArguments(node)
    # print(ast.dump(node))

class AssignmentVisitor(ast.NodeVisitor):
    def __init__(self):
        self.assignment_dictionary = {}

    def visit_Assign(self, node: Assign):
        # print(unparse(node))
        # print(ast.dump(node))
        assignment_target = node.targets[0].id
        assignment_value = node.value
        # print(assignment_target)
        # print(unparse(assignment_value))
        separate_api_parameter(assignment_value)
        api_call = node