import astunparse
import ast
from ast import *
# from astunparse import unparse

def print_code(node):
    print(astunparse.unparse(node))

# helper function to get the API invocation or function keyword arguments list
# return list of keyword argument name. e.g.
# getKeywordArguments(sklearn.cluster.KMeans(n_clusters=10).fit(X=[1,2,3])
# -> [X]
def getKeywordArguments(node):
    list_keyword = []
    if (isinstance(node, ast.Call)):
        for keyword in node.keywords:
            keyword_split = astunparse.unparse(keyword).split("=")
            keyword_string = keyword_split[0]
            list_keyword.append(keyword_string)
    return list_keyword


def getName(node):
    try:
        return node.attr
    except:
        try:
            return node.id
        except:
            try:
                return node.func.id
            except:
                try:
                    return node.func.attr
                except:
                    try:
                        return node.value.id
                    except:
                        return node.value.attr


# helper function to get the API invocation or function name from the node
# e.g.
# getFunctionName("sklearn.dummy.DummyClassifier") should return "DummyClassifier"
# getFunctionName("DummyClassifier") should also return "Dummy Classifier"
def getFunctionName(node):
    try:
        return node.func.id
    except:
        return node.func.attr

def changeFunctionName(node, newName):
    try:
        isExist = node.func.id
        node.func.id = newName
        return node
    except:
        node.func.attr = newName
        return node


def getScopeNode(node):
    try:
        return node.func.value
    except:
        try:
            return node.value
        except:
            return None

def recurseScope(node):
    returnList = []
    scope = getScopeNode(node)
    returnList.append(scope)
    if scope is not None:
        # Has scope that might be function call too
        recurseList = recurseScope(scope)
        returnList += recurseList
    return returnList


# helper function to get the API invocation or function scope
# e.g.
# getFunctionScope("sklearn.dummy.DummyClassifier") should return "sklearn.dummy"
# def getFunctionScope(node):
#     scope = ""
#     currNode = node.func
#     # First check whether it has scope. If it has scope, then the function node should be attribute
#     # else, the function node should be Name
#     if isinstance(currNode, _ast.Attribute):
#         currNode = currNode.value
#         while isinstance(currNode, _ast.Attribute):
#             if scope == "":
#                 scope = currNode.attr
#             else:
#                 scope = currNode.attr + "." + scope
#             try:
#                 currNode = currNode.value
#             except:
#                 break
#         if scope == "":
#             scope = currNode.id
#         else:
#             scope = currNode.id + "." + scope
#         return scope
#     else:
#         return None

# helper function to create a keyword param with a given name and value
# # keywords=[keyword(arg='strategy', value=Constant(value='stratified', kind=None))
def createKeywordParam(name, value):
    return ast.keyword(arg=name, value=Constant(value=value, kind=None))

def getOuterMostApi(node):
    # print("Full node: " + astunparse.unparse(node))
    while getScopeNode(node) is not None:
        node = getScopeNode(node)
    # print(astunparse.unparse(node))
    return node


