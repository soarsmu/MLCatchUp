import ast
from pprint import pprint

# Documentation for Python AST in the following links:
# 1. https://greentreesnakes.readthedocs.io/en/latest/index.html
# 2. https://kite.com/python/docs/ast.NodeVisitor



class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {"import": [], "from": []}

    def visit_Import(self, node):
        for alias in node.names:
            self.stats["import"].append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.stats["from"].append(alias.name)
        self.generic_visit(node)

    def report(self):
        pprint(self.stats)


with open("ast_test.py", "r") as source:
    tree = ast.parse(source.read())

analyzer = Analyzer()
analyzer.visit(tree)
analyzer.report()