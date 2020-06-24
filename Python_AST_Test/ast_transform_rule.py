import ast
import _ast
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

    def remove_param(self, node: Call):
        nodeFuncName = getFunctionName(node)
        if nodeFuncName == self.functionName:
            # Function name is correct
            listKeywordParam = getKeywordArguments(node)
            for keyword in listKeywordParam:
                # print(keyword.arg)
                if keyword.arg == self.parameterName:
                    listKeywordParam.remove(keyword)


    def visit_Call(self, node: Call):
        nodeFuncName = getFunctionName(node)
        self.remove_param(node)
        listScope = recurseScope(node)
        print(listScope)
        for n in listScope:
            if isinstance(n, _ast.Call):
                self.remove_param(n)
        return node

    def transform(self, tree):
        print_code(tree)

        self.visit(tree)

        print_code(tree)


# Class to process the change of API parameter default value
#
# dummy_clf = DummyClassifier() (Default value is strategy="stratified"))
#  ->
# dummy_clf = DummyClassifier(strategy="stratified")
class DefaultParamValueTransformer(ast.NodeTransformer):
    functionName = ""
    parameterName = ""
    oldDefaultValue = ""

    def __init__(self, fname, pname, oldvalue):
        self.functionName = fname
        self.parameterName = pname
        self.oldDefaultValue = oldvalue
        super().__init__()

    def default_value_transform(self, node: Call):
        nodeFuncName = getFunctionName(node)
        if nodeFuncName == self.functionName:
            # Function name is correct
            listKeywordParam = getKeywordArguments(node)
            isKeywordExist = False
            for keyword in listKeywordParam:
                # print(ast.dump(keyword))
                # print(keyword.arg)
                if keyword.arg == self.parameterName:
                    isKeywordExist = True
            if not isKeywordExist:
                # If keyword is not exist yet, it means that the old API use the default value which is changed
                # in the new API. Therefore, we need to create a new node
                # print("Keyword not exist")
                newParam = createKeywordParam(self.parameterName, self.oldDefaultValue)
                listKeywordParam.append(newParam)

    def visit_Call(self, node: Call):
        nodeFuncName = getFunctionName(node)
        self.default_value_transform(node)
        listScope = recurseScope(node)
        print(listScope)
        for n in listScope:
            if isinstance(n, _ast.Call):
                self.default_value_transform(n)
        return node

    def transform(self, tree):
        print_code(tree)
        self.visit(tree)
        # print(ast.dump(tree))
        print_code(tree)


class ApiNameTransformer(ast.NodeTransformer):
    functionName = ""
    newApiName = ""

    def __init__(self, fname, newname):
        self.functionName = fname
        self.newApiName = newname
        super().__init__()

    def name_transformer(self, node: Call):
        nodeFuncName = getFunctionName(node)
        if nodeFuncName == self.functionName:
            # Function name is correct
            node = changeFunctionName(node, self.newApiName)

    def visit_Call(self, node: Call):
        nodeFuncName = getFunctionName(node)
        self.name_transformer(node)
        listScope = recurseScope(node)
        print(listScope)
        for n in listScope:
            if isinstance(n, _ast.Call):
                self.name_transformer(n)
        return node

    def transform(self, tree):
        # print_code(tree)
        self.visit(tree)
        print_code(tree)