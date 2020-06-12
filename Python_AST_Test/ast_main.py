import ast
from ast import *
from pprint import pprint
import astunparse

# Documentation for Python AST in the following links:
# 1. https://greentreesnakes.readthedocs.io/en/latest/index.html
# 2. https://kite.com/python/docs/ast.NodeVisitor

# Things to remember when creating a new node:
# 1. A node must have lineno and col_offset attributes

# Some of the hurdles in AST modification in Python that do not exist in Java:
# 1. Dynamic typing
# 2. No usage of semicolon that can be used to determine a statement ending
# 3. No usage of brackets that are instead changed into indentation. These indentation makes it harder to create
#    a new node because we will need to determine the line number and the offset of such new node.
from Python_AST_Test.ast_utils import *
from Python_AST_Test.ast_transform_rule import *



with open("ast_test.py", "r") as source:
    tree = ast.parse(source.read())
paramRemove = KeywordParamRemover("KMeans", "n_clusters")
paramRemove.transform(tree)

paramRemove = KeywordParamRemover("KMeans", "n_jobs")
paramRemove.transform(tree)

defaultValueChange = DefaultParamValueTransformer()