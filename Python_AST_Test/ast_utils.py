import astunparse
import ast
from ast import *
def print_code(node):
    print(astunparse.unparse(node))

# helper function to get the API invocation or function name from the node
# e.g.
# getFunctionName("sklearn.dummy.DummyClassifier") should return "DummyClassifier"
# getFunctionName("DummyClassifier") should also return "Dummy Classifier"
def getFunctionName(node):
    print("\n")
    print("\n")
    print_code(node)
    print(ast.dump(node))
    try:
        return node.func.id
    except:
        return node.func.attr

# helper function to get the API invocation or function scope
# e.g.
# getFunctionScope("sklearn.dummy.DummyClassifier") should return "sklearn.dummy"
def getFunctionScope(node):



def getFunctionArguments(node):
    return node