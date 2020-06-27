import ast
from astunparse import unparse
from ast import Assign
from Python_AST_Test.ast_utils import *

def separate_api_parameter(node):
    # Loop through the given node from assignment to get all the API Call and keyword
    api_name = getName(node)
    api_keywords = []
    api_keywords.extend(getKeywordArguments(node))
    scope_list = recurseScope(node)
    for scope in scope_list:
        if scope is not None:
            api_keywords.extend(getKeywordArguments(scope))
            api_name = getName(scope) + "." + api_name
    separated_signature = api_name + "#"
    last_element = api_keywords[:-1]
    for key in api_keywords:
        separated_signature += key
        if key != last_element:
            separated_signature += ", "
    return api_name, api_keywords

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
        api_name, api_keywords = separate_api_parameter(assignment_value)

        dict_content = {}
        dict_content["name"] = api_name
        dict_content["key"] = api_keywords
        dict_content["line_no"] = node.lineno
        self.assignment_dictionary[assignment_target] = []
        self.assignment_dictionary[assignment_target].append(dict_content)
        #
        # print("Assignment target: " + assignment_target)
        # print("Assignment visit: " + unparse(node))
        # print("Name visit: " + self.assignment_dictionary[assignment_target][0]["name"])
        # print("Keywords visit: " + self.assignment_dictionary[assignment_target][0]["key"].__str__())
        # print("Line Number: " + self.assignment_dictionary[assignment_target][0]["line_no"].__str__())

class ImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.import_dictionary = {}

    def visit_Import(self, node):


        import_path = node.names[0].name
        import_alias = node.names[0].asname
        if import_alias is not None:
            self.import_dictionary[import_alias] = import_path

class FromImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.from_import_dict = {}

    def visit_ImportFrom(self, node: ImportFrom):

        module_name = node.module
        imported_name = node.names[0].name
        imported_alias = node.names[0].asname
        print("Module name: " + module_name)
        print("Imported name: " + imported_name)
        print("Imported alias: " + imported_alias.__str__())
        full_path = module_name + "." + imported_name
        print("Full name: " + full_path)
        if imported_alias is not None:
            print()
        else:
            print()


# Tree is the AST of the complete file
def process_api_format(tree):
    # assignmentVisitor = AssignmentVisitor()
    # assignmentVisitor.visit(tree)
    # assign_dict = assignmentVisitor.assignment_dictionary
    # print(assign_dict.__str__())

    importVisitor = ImportVisitor()
    importVisitor.visit(tree)

    fromImportVisitor = FromImportVisitor()
    fromImportVisitor.visit(tree)