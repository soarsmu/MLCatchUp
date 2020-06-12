import astunparse
import ast
import _ast
from ast import *
def print_code(node):
    print(astunparse.unparse(node))

# helper function to get the API invocation or function name from the node
# e.g.
# getFunctionName("sklearn.dummy.DummyClassifier") should return "DummyClassifier"
# getFunctionName("DummyClassifier") should also return "Dummy Classifier"
def getFunctionName(node):
    try:
        return node.func.id
    except:
        return node.func.attr

# helper function to get the API invocation or function scope
# e.g.
# getFunctionScope("sklearn.dummy.DummyClassifier") should return "sklearn.dummy"
def getFunctionScope(node):
    scope = ""
    currNode = node.func
    # First check whether it has scope. If it has scope, then the function node should be attribute
    # else, the function node should be Name
    if isinstance(currNode, _ast.Attribute):
        currNode = currNode.value
        while isinstance(currNode, _ast.Attribute):
            if scope == "":
                scope = currNode.attr
            else:
                scope = currNode.attr + "." + scope
            try:
                currNode = currNode.value
            except:
                break
        if scope == "":
            scope = currNode.id
        else:
            scope = currNode.id + "." + scope
        return scope
    else:
        return None


# helper function to get the API invocation or function keyword arguments list
def getKeywordArguments(node):
    return node.keywords