import ast
from ast import *
from Python_AST_Test.ast_utils import *

# Class to remove parameter from an API invocation
class KeywordParamRemover(ast.NodeTransformer):
    functionName = ""
    parameterName = ""

    def __init__(self, fname, pname):
        self.functionName = fname
        self.parameterName = pname
        super().__init__()

    def visit_Call(self, node: Call):
        nodeFunction = node.func
        nodeFuncName = node.func.id
        # print(ast.dump(node))
        if nodeFuncName == self.functionName:
            # Function name is correct
            listKeywordParam = node.keywords
            print(listKeywordParam)
            for keyword in listKeywordParam:
                # print(keyword.arg)
                if keyword.arg == self.parameterName:
                    listKeywordParam.remove(keyword)
                    keyword = None
        return node

    def transform(self, tree):
        print_code(tree)
        self.visit(tree)
        # print(ast.dump(tree))
        print_code(tree)