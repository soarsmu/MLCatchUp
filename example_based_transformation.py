import ast
import _ast
from ast import *
from ast_utils import *
from astunparse import unparse
import re


# Class to remove parameter from an API invocation
class GetFirstCallNode(ast.NodeVisitor):
    functionName = ""
    first_node = None
    list_line_number = []

    def __init__(self, fname, listlinenumber):
        self.functionName = fname
        self.list_line_number = listlinenumber
        super().__init__()

    def visit_Call(self, node: Call):
        if node.lineno in self.list_line_number and self.first_node is None:
            self.first_node = node

    def get(self, tree):
        self.listChanges = []
        self.visit(tree)
        return self.first_node
        # print("Updated code: ")
        # print_code(tree)
        # print("----------------------------------------------------------------------------------------------------")
