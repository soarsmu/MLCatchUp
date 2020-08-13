import ast
import _ast
from ast import *
from ast_utils import *
from astunparse import unparse
from api_formatter import *
from input_utils import ApiSignature
import re

# Helper function to get the list of line number
def get_list_API(tree, api_signature: ApiSignature):
    api_name = api_signature.api_name
    list_api, import_dict, from_import_dict = process_api_format(tree, api_name)
    list_deprecated_api = []
    list_completed_api = []
    for api in list_api:
        if api_name.strip() == api["name"].strip():
            key_is_correct = True
            list_deprecated_api.append(api)

    for api in list_deprecated_api:
        print("This is API")
        print(api)
        list_completed_api.append(api)
    return list_completed_api

def get_list_line_number(list_completed_api):
    list_line_number = []
    for api in list_completed_api:
        list_line_number.append(api["line_no"])
    return list_line_number

# Class to remove parameter from an API invocation
class KeywordParamRemover(ast.NodeTransformer):
    functionName = ""
    parameterName = ""
    listChanges = []
    list_line_number = []

    def __init__(self, fname, pname, listlinenumber):
        self.functionName = fname
        self.parameterName = pname
        self.list_line_number = listlinenumber
        super().__init__()

    def remove_param(self, node: Call):
        # Function name is correct
        # This first one is easy check to make sure that there is a relevant keyword here
        listKeywordParam = getKeywordArguments(node)
        for keyword in listKeywordParam:
            # print(keyword.arg)
            if keyword == self.parameterName:
                keyword_ast = node.keywords
                for key_ast in keyword_ast:
                    if key_ast.arg == self.parameterName:
                        keyword_ast.remove(key_ast)
                node.keywords = keyword_ast


    def visit_Call(self, node: Call):
        if node.lineno in self.list_line_number:
            self.listChanges.append("Deprecated API detected in line: " + node.lineno.__str__())
            self.listChanges.append("Content: \n" + unparse(node))
            tempString = ""
            self.remove_param(node)
            listScope = recurseScope(node)
            for n in listScope:
                if isinstance(n, _ast.Call):
                    self.remove_param(n)
            self.listChanges.append("Updated content: \n" + unparse(node))
        return node

    def transform(self, tree):
        self.listChanges = []
        self.visit(tree)
        # print("Updated code: ")
        # print_code(tree)
        # print("----------------------------------------------------------------------------------------------------")

# Class to change keyword parameter
class KeywordParamChanger(ast.NodeTransformer):
    functionName = ""
    parameterName = ""
    listChanges = []
    list_line_number = []

    def __init__(self, fname, pname, new_param_name, listlinenumber):
        self.functionName = fname
        self.parameterName = pname
        self.new_param_name = new_param_name
        self.list_line_number = listlinenumber
        super().__init__()

    def change_param(self, node: Call):
        # Function name is correct
        # This first one is easy check to make sure that there is a relevant keyword here
        listKeywordParam = getKeywordArguments(node)
        for keyword in listKeywordParam:
            # print(keyword.arg)
            if keyword == self.parameterName:
                keyword_ast = node.keywords
                for key_ast in keyword_ast:
                    if key_ast.arg == self.parameterName:
                        new_keyword = key_ast
                        new_keyword.arg = self.new_param_name
                        keyword_ast.remove(key_ast)
                        keyword_ast.append(new_keyword)
                node.keywords = keyword_ast

    def visit_Call(self, node: Call):
        if node.lineno in self.list_line_number:
            self.listChanges.append("Deprecated API detected in line: " + node.lineno.__str__())
            self.listChanges.append("Content: \n" + unparse(node))
            tempString = ""
            self.change_param(node)
            listScope = recurseScope(node)
            for n in listScope:
                if isinstance(n, _ast.Call):
                    self.change_param(n)
            self.listChanges.append("Updated content: \n" + unparse(node))
        return node

    def transform(self, tree):
        self.listChanges = []
        self.visit(tree)
        # print("Updated code: ")
        print_code(tree)
        # print("----------------------------------------------------------------------------------------------------")

class ApiNameTransformer(ast.NodeTransformer):
    functionName = ""
    newApiName = ""

    def __init__(self, fname, newname, list_line_number, list_found_api):
        self.list_line_number = list_line_number
        self.oldApiName = fname
        self.newApiName = newname
        self.listChanges = []
        self.found_api = list_found_api
        super().__init__()

    def visit_Call(self, node: Call):
        if node.lineno in self.list_line_number:
            actual_api = {}
            for api in self.found_api:
                # found the actual api
                if api["line_no"] == node.lineno:
                    actual_api = api

            self.listChanges.append("Deprecated API detected in line: " + node.lineno.__str__())
            self.listChanges.append("Content: \n" + unparse(node))
            print("Deprecated API detected in line: " + node.lineno.__str__())
            print("Content: \n" + unparse(node))

            convert_all = True
            api_without_arguments = unparse(node)
            nb_rep = 1
            # Remove the arguments
            while (nb_rep):
                (api_without_arguments, nb_rep) = re.subn(r'\([^()]*\)', '', api_without_arguments)


            excessAPI = actual_api['name'].replace(self.oldApiName, '')

            api_without_arguments = api_without_arguments.replace(excessAPI, '')

            if not self.change_whole and len(api_without_arguments.split(".")) > 1:
                # Should also process excess API here?
                # Change the whole API invocation
                # Find if there are excess API invocation / object

                # Case have excess API (e.g. KMeans that is followed by .fit)

                currentApi = unparse(node)
                if len(excessAPI) > 1:
                    first_part_excess = excessAPI.split(".")[1]
                    idx = currentApi.index(first_part_excess)
                    # changed part contain the API invocation without the excess invocation
                    changed_part = currentApi[0:idx - 1]
                else:
                    changed_part = currentApi
                print(changed_part)
                changed_part_without_arg = changed_part
                nb_rep = 1
                while (nb_rep):
                    (changed_part_without_arg, nb_rep) = re.subn(r'\([^()]*\)', '', changed_part_without_arg)
                function_name_only = changed_part_without_arg.split(".")[-1].strip()
                print("THIS IS FUNCTION_NAME ONLY: " + function_name_only)
                print("THIS IS CHANGED PART NEW: " + changed_part)
                index_of_split = changed_part.index(function_name_only)
                prepend_api = changed_part[0:index_of_split]
                need_to_be_modified_api = changed_part[index_of_split:]
                print("THIS IS PREPEND API: " + prepend_api)
                print("THIS IS NEED TO BE MODIFIED: " + need_to_be_modified_api)
                bracket_index = need_to_be_modified_api.index("(")
                modified_name = need_to_be_modified_api[0:bracket_index]
                modified_argument = need_to_be_modified_api[bracket_index:]
                print("MODIFIED NAME: " + modified_name)
                print("MODIFIED ARGUMENT: " + modified_argument)
                modified_name = self.newApiName.split(".")[-1]


                if len(excessAPI) > 1:
                    print("Must process the excess API too")
                    print("TRY AST PARSE")
                    # String processing

                    # Index before the excess API
                    first_part_excess = excessAPI.split(".")[1]
                    idx = currentApi.index(first_part_excess)
                    # changed part contain the API invocation without the excess invocation
                    changed_part = currentApi[0:idx - 1]
                    excess_part = currentApi.replace(changed_part, '')
                    # Find out what is the last part argument by using regex

                    newApi = prepend_api + modified_name + modified_argument + excess_part
                    print("NEW API: " + newApi)
                    parsed_code = ast.parse(newApi, mode="eval")
                    call_node = parsed_code.body
                    node = call_node
                else:
                    newApi = prepend_api + modified_name + modified_argument
                    print("NEW API: " + newApi)
                    parsed_code = ast.parse(newApi, mode="eval")
                    call_node = parsed_code.body
                    node = call_node
            else:
                self.need_to_add_import = True
                # Change the whole API invocation
                # Find if there are excess API invocation / object
                excessAPI = actual_api['name'].replace(self.oldApiName, '')
                # Case have excess API (e.g. KMeans that is followed by .fit)
                if len(excessAPI) > 1:
                    print("Must process the excess API too")
                    print("TRY AST PARSE")
                    # String processing
                    currentApi = unparse(node)
                    # Index before the excess API
                    first_part_excess = excessAPI.split(".")[1]
                    idx = currentApi.index(first_part_excess)
                    # changed part contain the API invocation without the excess invocation
                    changed_part = currentApi[0:idx - 1]
                    excess_part = currentApi.replace(changed_part, '')
                    # Find out what is the last part argument by using regex
                    last_part = changed_part
                    nb_rep = 1
                    while (nb_rep):
                        (last_part, nb_rep) = re.subn(r'\([^()]*\)', '', last_part)
                    last_part = last_part.split(".")[-1]
                    api_arguments = changed_part.split(last_part)[1]
                    newApi = self.newApiName.split(".")[-1] + api_arguments + excess_part
                    parsed_code = ast.parse(newApi, mode="eval")
                    call_node = parsed_code.body
                    node = call_node
                else:
                    print("Simply get the arguments and create new API invocations")
                    positional_arg = node.args
                    keyword_arg = node.keywords
                    context = node.func.ctx
                    newInvocation = ast.Call(func=Name(id=self.newApiName.split(".")[-1], ctx=context), args=positional_arg, keywords=keyword_arg)
                    node = newInvocation


            tempString = ""
            # self.name_transformer(node)
            # listScope = recurseScope(node)
            # for n in listScope:
            #     if isinstance(n, _ast.Call):
            #         self.name_transformer(n)
            # self.listChanges.append("Updated content: \n" + unparse(node))
        return node

    def transform(self, tree):
        # print_code(tree)


        # First check if the change is only at the end or also change the API parent object/class
        # Compare the oldapiname and the newapiname
        split_old = self.oldApiName.split(".")
        split_new = self.newApiName.split(".")
        isParentSame = True
        len_old = len(split_old)
        len_new = len(split_new)
        if len_old != len_new:
            isParentSame = False
        else:
            for i in range(0, len_old - 1):
                if split_old[i] != split_new[i]:
                    isParentSame = False
                    break
        # Check the function name
        if split_old[-1] == split_new[-1]:
            isMethodNameSame = True
        else:
            isMethodNameSame = False

        print("isParentSame: " + isParentSame.__str__())
        print("isMethodSame: " + isMethodNameSame.__str__())

        # Case parent same and method not same = only change the end name of the function
        # Case parent diff = better be safe and create new import and change the whole API invocation
        # This will cause a bug if the parent contains other API invocations (which is unlikely if the parent is changed)


        # Check whether there is any deprecated API first
        # For now, it does not matter whether only the function is change or the fully qualified API name is changed

        # It may cause a bug in case the invoked API is dependent on the parent object/attribute
        # Therefore, if the change is only on the function name which is a part of other API invocation (i.e. not
        # a standalone API invocation, we should just change the name of the API without considering the import
        # since the import will still be exist)

        if len(self.list_line_number) > 0:
            if not isMethodNameSame and isParentSame:
                self.change_whole = False
            else:
                self.change_whole = True
            # Change the whole function
            # Will need to change the import!
            print("change whole function")

            # new parent name
            parent_name = ".".join(split_new[0:-1])
            new_api_name = split_new[-1]

            for node in ast.walk(tree):
                if type(node) == ast.ImportFrom:
                    print(node)
                    print(ast.dump(node))

            # Create the new import statement first
            import_node = ast.ImportFrom(module=parent_name, names=[alias(name=new_api_name, asname=None)], level=0)

            print(ast.dump(tree))
            # add the new import just before the first API invocation for now
            # TODO: Think of better placement of the new import
            print("THIS IS LIST LINE NUMBER")
            print(self.list_line_number.__str__())


            # TODO: Change the API invocation into the new imported name (i.e. new_api_name)
            self.need_to_add_import = False
            self.visit(tree)
            if self.need_to_add_import:
                tree.body.insert(self.list_line_number[0] - 1, import_node)


            print_code(tree)


        # self.visit(tree)
        # print_code(tree)

# Convert positional parameter to keyword parameter
class PositionalToKeyword(ast.NodeTransformer):
    functionName = ""
    parameterName = ""
    listChanges = []
    list_line_number = []

    def __init__(self, api_name, param_position, new_keyword, list_line_number):
        self.functionName = api_name
        self.parameter_position = param_position
        self.parameter_keyword = new_keyword
        self.list_line_number = list_line_number
        super().__init__()

    def positional_to_keyword(self, node: Call):
        # Function name is correct

        # PERHAPS FOR THIS WE NEED TO MAKE SURE THAT THE POSITIONAL ARGUMENT IS OF THE CORRECT TYPE
        # E.G. MAKE SURE THAT THE SECOND POSITIONAL ARGUMENT IS INTEGER TYPE
        # OR MAYBE WE ALSO NEED TO MAKE SURE THE NUMBER OF ARGUMENT IS CORRECT (E.G. 3 POSITIONAL PARAMETER)
        # THIS WILL BE A FUTURE TODO
        # CHECKING THE TYPE OF THE POSITIONAL ARGUMENT WILL BE HARDER SINCE IT CAN BE A NAME, API CALL, CONSTANT, ETC

        listPositionalParam = node.args
        # print(listPositionalParam.__str__())
        print("TODO TODO TODO")
        print(ast.dump(node))
        if len(listPositionalParam) >= int(self.parameter_position):
            # since the parameter position start from 1 while the index start from 0
            value_args = listPositionalParam.pop(int(self.parameter_position) - 1)
            listKeyword = node.keywords
            # Create the new keyword
            new_keyword = ast.keyword(arg=self.parameter_keyword, value=value_args)
            listKeyword.append(new_keyword)
            node.args = listPositionalParam
            node.keywords = listKeyword

        # listKeywordParam = getKeywordArguments(node)
        #
        # for keyword in listKeywordParam:
        #     # print(keyword.arg)
        #     if keyword == self.parameterName:
        #         keyword_ast = node.keywords
        #         for key_ast in keyword_ast:
        #             if key_ast.arg == self.parameterName:
        #                 new_keyword = key_ast
        #                 new_keyword.arg = self.new_param_name
        #                 keyword_ast.remove(key_ast)
        #                 keyword_ast.append(new_keyword)
        #         node.keywords = keyword_ast



    def visit_Call(self, node: Call):
        if node.lineno in self.list_line_number:
            self.listChanges.append("Deprecated API detected in line: " + node.lineno.__str__())
            self.listChanges.append("Content: \n" + unparse(node))
            self.positional_to_keyword(node)
            listScope = recurseScope(node)
            for n in listScope:
                if isinstance(n, _ast.Call):
                    # Check if the keyword is already present
                    isExist = False
                    listKeyword = node.keywords
                    for key in listKeyword:
                        if key.arg == self.parameterName:
                            isExist = True
                            break
                    if not isExist:
                        self.positional_to_keyword(n)
            self.listChanges.append("Updated content: \n" + unparse(node))
        return node

    def transform(self, tree):
        self.listChanges = []
        self.visit(tree)
        # print("Updated code: ")
        print_code(tree)
        # print("----------------------------------------------------------------------------------------------------")

# Remove positional parameter
# TODO - NOT DONE YET
class PositionalParamRemover(ast.NodeTransformer):
    functionName = ""
    listChanges = []
    list_line_number = []

    def __init__(self, api_name, param_position, new_keyword, list_line_number):
        self.functionName = api_name
        self.parameter_position = param_position
        self.parameter_keyword = new_keyword
        self.list_line_number = list_line_number
        super().__init__()

    def remove_positional_param(self, node: Call):
        # Function name is correct
        # PERHAPS FOR THIS WE NEED TO MAKE SURE THAT THE POSITIONAL ARGUMENT IS OF THE CORRECT TYPE
        # E.G. MAKE SURE THAT THE SECOND POSITIONAL ARGUMENT IS INTEGER TYPE
        # OR MAYBE WE ALSO NEED TO MAKE SURE THE NUMBER OF ARGUMENT IS CORRECT (E.G. 3 POSITIONAL PARAMETER)
        # THIS WILL BE A FUTURE TODO
        # CHECKING THE TYPE OF THE POSITIONAL ARGUMENT WILL BE HARDER SINCE IT CAN BE A NAME, API CALL, CONSTANT, ETC

        listPositionalParam = node.args
        # print(listPositionalParam.__str__())
        print("TODO TODO TODO")
        print(ast.dump(node))
        if len(listPositionalParam) >= self.parameter_position:
            # since the parameter position start from 1 while the index start from 0
            listPositionalParam.pop(self.parameter_position - 1)
            node.args = listPositionalParam

    def visit_Call(self, node: Call):
        if node.lineno in self.list_line_number:
            self.listChanges.append("Deprecated API detected in line: " + node.lineno.__str__())
            self.listChanges.append("Content: \n" + unparse(node))
            self.remove_positional_param(node)
            listScope = recurseScope(node)
            for n in listScope:
                if isinstance(n, _ast.Call):
                    self.remove_positional_param(n)
            self.listChanges.append("Updated content: \n" + unparse(node))
        return node

    def transform(self, tree):
        self.listChanges = []
        self.visit(tree)
        # print("Updated code: ")
        print_code(tree)
        # print("----------------------------------------------------------------------------------------------------")


# The parameterValue that can be passed is limitted?
# E.g. String, Numeral, Boolean
class AddNewParameter(ast.NodeTransformer):
    functionName = ""
    parameterName = ""
    listChanges = []
    list_line_number = []

    def __init__(self, api_name, parameter_name, parameter_type, parameter_value, list_line_number):
        self.functionName = api_name
        self.parameterName = parameter_name
        self.parameterType = parameter_type
        self.parameterValue = parameter_value

        self.list_line_number = list_line_number
        super().__init__()

    def add_new_parameter(self, node: Call):
        # print(node)
        # print(ast.dump(node))
        # print(unparse(node))
        # Call
        # Attribute
        # Name
        # Constant
        if self.parameterType.lower() == "call":
            # API call or object instantiation
            # TODO: Need a better call parser (i.e. separate between the name and the arguments of the call)
            print("Call type")
            try:
                keyword_value = ast.parse(self.parameterValue).body[0].value
            except Exception as e:
                print(e)
                exit(1)
        elif self.parameterType.lower() == "attribute":
            # class or object or package attribute type
            # TODO: Need a better parser for the attribute too (i.e. case if recursive attribute)
            print("Attribute type")
            # new_value
            try:
                keyword_value = ast.parse(self.parameterValue).body[0].value
            except Exception as e:
                print(e)
                exit(1)
        elif self.parameterType.lower() == "name":
            # name or variable type
            print("Name type")
            try:
                keyword_value = ast.parse(self.parameterValue).body[0].value
            except Exception as e:
                print(e)
                exit(1)
        elif self.parameterType.lower() == "constant":
            # constant type
            print("Constant type")
            keyword_value = ast.Constant(value=self.parameterValue, kind=None)
        else:
            print("Broken input")
            # exit(1)
        new_keyword = ast.keyword(arg=self.parameterName, value=keyword_value)
        node.keywords.append(new_keyword)
        return node

    def visit_Call(self, node: Call):
        if node.lineno in self.list_line_number:
            self.listChanges.append("Deprecated API detected in line: " + node.lineno.__str__())
            self.listChanges.append("Content: \n" + unparse(node))
            self.add_new_parameter(node)
            listScope = recurseScope(node)
            for n in listScope:
                if isinstance(n, _ast.Call):
                    # Check if the keyword is already present
                    isExist = False
                    listKeyword = node.keywords
                    for key in listKeyword:
                        if key.arg == self.parameterName:
                            isExist = True
                            break
                    if not isExist:
                        self.add_new_parameter(n)
            self.listChanges.append("Updated content: \n" + unparse(node))
        return node

    def transform(self, tree):
        self.listChanges = []
        self.visit(tree)
        # print("Updated code: ")
        print_code(tree)
        # print("----------------------------------------------------------------------------------------------------")






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
        for n in listScope:
            if isinstance(n, _ast.Call):
                self.default_value_transform(n)
        return node

    def transform(self, tree):
        # print_code(tree)
        self.visit(tree)
        # print(ast.dump(tree))
        print_code(tree)
