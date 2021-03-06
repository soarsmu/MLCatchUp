import ast
import _ast
from ast import *
from ast_utils import *
from astunparse import unparse
from api_formatter import *
from input_utils import ApiSignature
import re

class ChangeNode:
    def __init__(self):
        pass

# Helper function to get the list of line number
def get_list_API(tree, api_signature: ApiSignature):
    api_name = api_signature.api_name
    list_api, import_dict, from_import_dict = process_api_format(tree, api_name)
    list_completed_api = []
    for api in list_api:
        if api_name.strip() in api["name"].strip():
            # temp = api["name"].strip().replace(api_name.strip, '')
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
    dict_change = {}
    list_completed_API = []

    def __init__(self, fname, pname, listlinenumber, list_completed_API):
        self.functionName = fname
        self.parameterName = pname
        self.list_line_number = listlinenumber
        self.dict_change = {}
        self.list_completed_API = list_completed_API
        super().__init__()

    def remove_param(self, node: Call):
        # Function name is correct
        # This first one is easy check to make sure that there is a relevant keyword here
        listKeywordParam = getKeywordArguments(node)
        for keyword in listKeywordParam:
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
                # New version that check the scope name
                if isinstance(n, _ast.Call) and self.functionName[self.functionName.rfind(".") + 1:] in unparse(n):
                # Original version that does not check the scope name
                # if isinstance(n, _ast.Call):
                    self.remove_param(n)
            self.listChanges.append("Updated content: \n" + unparse(node))
            self.dict_change[node.lineno] = node
        return node

    def transform(self, tree):
        self.listChanges = []
        self.visit(tree)
        return self.dict_change

# Class to change keyword parameter
class KeywordParamChanger(ast.NodeTransformer):
    functionName = ""
    parameterName = ""
    listChanges = []
    list_line_number = []
    dict_change = {}
    list_completed_API =[]

    def __init__(self, fname, pname, new_param_name, listlinenumber, list_completed_API):
        self.functionName = fname
        self.parameterName = pname
        self.new_param_name = new_param_name
        self.list_line_number = listlinenumber
        self.dict_change = {}
        self.list_completed_API = list_completed_API
        super().__init__()

    def change_param(self, node: Call):
        # Function name is correct
        # This first one is easy check to make sure that there is a relevant keyword here
        listKeywordParam = getKeywordArguments(node)
        for keyword in listKeywordParam:
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
                if isinstance(n, _ast.Call) and self.functionName[self.functionName.rfind(".") + 1:] in unparse(n):
                    self.change_param(n)
            self.listChanges.append("Updated content: \n" + unparse(node))
            self.dict_change[node.lineno] = node
        return node

    def transform(self, tree):
        self.listChanges = []
        self.visit(tree)
        return self.dict_change

class ApiNameTransformer(ast.NodeTransformer):
    functionName = ""
    newApiName = ""
    dict_change = {}
    list_completed_API = []

    def __init__(self, fname, newname, list_line_number, list_found_api):
        self.list_line_number = list_line_number
        self.oldApiName = fname
        self.newApiName = newname
        self.listChanges = []
        self.found_api = list_found_api
        self.dict_change = {}
        self.list_completed_API = list_found_api
        super().__init__()

    def name_changer(self, node: Call):
        actual_node_pos = node.lineno
        actual_api = {}
        for api in self.found_api:
            # found the actual api
            if api["line_no"] == node.lineno:
                actual_api = api

        self.listChanges.append("Deprecated API detected in line: " + node.lineno.__str__())
        self.listChanges.append("Content: \n" + unparse(node))
        convert_all = True
        api_without_arguments = unparse(node)
        nb_rep = 1
        # Remove the arguments
        while (nb_rep):
            (api_without_arguments, nb_rep) = re.subn(r'\([^()]*\)', '', api_without_arguments)

        # excessAPI = actual_api['name'].replace(self.oldApiName, '')
        excessAPI = ""

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

            changed_part_without_arg = changed_part
            nb_rep = 1
            while (nb_rep):
                (changed_part_without_arg, nb_rep) = re.subn(r'\([^()]*\)', '', changed_part_without_arg)
            function_name_only = changed_part_without_arg.split(".")[-1].strip()

            index_of_split = changed_part.index(function_name_only)
            prepend_api = changed_part[0:index_of_split]
            need_to_be_modified_api = changed_part[index_of_split:]

            bracket_index = need_to_be_modified_api.index("(")
            modified_name = need_to_be_modified_api[0:bracket_index]
            modified_argument = need_to_be_modified_api[bracket_index:]

            modified_name = self.newApiName.split(".")[-1]

            if len(excessAPI) > 1:
                # String processing

                # Index before the excess API
                first_part_excess = excessAPI.split(".")[1]
                idx = currentApi.index(first_part_excess)
                # changed part contain the API invocation without the excess invocation
                changed_part = currentApi[0:idx - 1]
                excess_part = currentApi.replace(changed_part, '')
                # Find out what is the last part argument by using regex

                newApi = prepend_api + modified_name + modified_argument + excess_part
                parsed_code = ast.parse(newApi, mode="eval")
                call_node = parsed_code.body
                node = call_node
            else:
                newApi = prepend_api + modified_name + modified_argument
                parsed_code = ast.parse(newApi, mode="eval")
                call_node = parsed_code.body
                node = call_node
        else:
            self.need_to_add_import = True
            # Change the whole API invocation
            # Find if there are excess API invocation / object
            # excessAPI = actual_api['name'].replace(self.oldApiName, '')
            excessAPI = ""
            # Case have excess API (e.g. KMeans that is followed by .fit)
            if len(excessAPI) > 1:
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
                positional_arg = node.args
                keyword_arg = node.keywords
                context = node.func.ctx
                newInvocation = ast.Call(func=Name(id=self.newApiName.split(".")[-1], ctx=context), args=positional_arg,
                                         keywords=keyword_arg)
                node = newInvocation

        # Hard code way to set the line position of the node
        return node

    def visit_Call(self, node: Call):
        # Add extra check for the function name
        last_part_oldname = self.oldApiName[self.oldApiName.rfind(".") + 1:]
        if node.lineno in self.list_line_number and last_part_oldname in unparse(node):
            idx = self.list_line_number.index(node.lineno)
            api_object = self.list_completed_API[idx]
            if api_object["type"] == "ARGUMENT":
                call_node = ast.parse(api_object["code"]).body[0].value
                updated_node = self.name_changer(call_node)
                original_code = unparse(node).rstrip()
                deprecated_call = api_object["code"].rstrip()
                updated_call = unparse(updated_node).rstrip()
                updated_code = original_code.replace(deprecated_call, updated_call)
                final_node = ast.parse(updated_code).body[0].value
                self.dict_change[node.lineno] = final_node
                # self.name_changer(node)
                return final_node
            else:
                actual_node_pos = node.lineno
                actual_api = {}
                for api in self.found_api:
                    # found the actual api
                    if api["line_no"] == node.lineno:
                        actual_api = api

                self.listChanges.append("Deprecated API detected in line: " + node.lineno.__str__())
                self.listChanges.append("Content: \n" + unparse(node))
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

                    changed_part_without_arg = changed_part
                    nb_rep = 1
                    while (nb_rep):
                        (changed_part_without_arg, nb_rep) = re.subn(r'\([^()]*\)', '', changed_part_without_arg)
                    function_name_only = changed_part_without_arg.split(".")[-1].strip()

                    index_of_split = changed_part.index(function_name_only)
                    prepend_api = changed_part[0:index_of_split]
                    need_to_be_modified_api = changed_part[index_of_split:]

                    bracket_index = need_to_be_modified_api.index("(")
                    modified_name = need_to_be_modified_api[0:bracket_index]
                    modified_argument = need_to_be_modified_api[bracket_index:]

                    modified_name = self.newApiName.split(".")[-1]


                    if len(excessAPI) > 1:
                        # String processing

                        # Index before the excess API
                        first_part_excess = excessAPI.split(".")[1]
                        idx = currentApi.index(first_part_excess)
                        # changed part contain the API invocation without the excess invocation
                        changed_part = currentApi[0:idx - 1]
                        excess_part = currentApi.replace(changed_part, '')
                        # Find out what is the last part argument by using regex

                        newApi = prepend_api + modified_name + modified_argument + excess_part
                        parsed_code = ast.parse(newApi, mode="eval")
                        call_node = parsed_code.body
                        node = call_node
                    else:
                        newApi = prepend_api + modified_name + modified_argument
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
                        positional_arg = node.args
                        keyword_arg = node.keywords
                        context = node.func.ctx
                        newInvocation = ast.Call(func=Name(id=self.newApiName.split(".")[-1], ctx=context), args=positional_arg, keywords=keyword_arg)
                        node = newInvocation

                # Hard code way to set the line position of the node
                node.lineno = actual_node_pos
                self.dict_change[actual_node_pos] = node

        return node

    def transform(self, tree):
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

            # new parent name
            parent_name = ".".join(split_new[0:-1])
            new_api_name = split_new[-1]

            # Create the new import statement first
            import_node = ast.ImportFrom(module=parent_name, names=[alias(name=new_api_name, asname=None)], level=0)

            self.need_to_add_import = False
            self.visit(tree)
            if self.need_to_add_import:
                tree.body.insert(0, import_node)
                self.dict_change[0] = import_node

            return self.dict_change
        else:
            return {}

# Remove API visitor
class RemoveAPI(ast.NodeTransformer):
    functionName = ""
    list_line_number = ""
    dict_change = {}
    list_completed_API = []

    def __init__(self, api_name, list_line_number, list_completed_API):
        self.functionName = api_name
        self.list_line_number = list_line_number
        self.dict_change = {}
        self.list_completed_API = list_completed_API

    def visit_Call(self, node: Call):
        if node.lineno in self.list_line_number:
            newInvocation = ast.Call(func=Name(id="EMPTYSHOULDBEDELETED"), args=[], keywords=[])
            original_lineNo= node.lineno
            node = newInvocation
            self.dict_change[original_lineNo] = newInvocation
        return node

    def transform(self, tree):
        self.visit(tree)
        return self.dict_change

# Convert positional parameter to keyword parameter
class PositionalToKeyword(ast.NodeTransformer):
    functionName = ""
    parameterName = ""
    listChanges = []
    list_line_number = []
    dict_change = {}
    list_completed_API = []

    def __init__(self, api_name, param_position, new_keyword, list_line_number, list_completed_API):
        self.functionName = api_name
        self.parameter_position = param_position
        self.parameter_keyword = new_keyword
        self.list_line_number = list_line_number
        self.dict_change = {}
        self.list_completed_API = list_completed_API
        super().__init__()

    def positional_to_keyword(self, node: Call):
        # Function name is correct
        listPositionalParam = node.args
        if len(listPositionalParam) >= int(self.parameter_position):
            # since the parameter position start from 1 while the index start from 0
            value_args = listPositionalParam.pop(int(self.parameter_position) - 1)
            listKeyword = node.keywords
            # Create the new keyword
            new_keyword = ast.keyword(arg=self.parameter_keyword, value=value_args)
            listKeyword.append(new_keyword)
            node.args = listPositionalParam
            node.keywords = listKeyword


    def visit_Call(self, node: Call):
        if node.lineno in self.list_line_number:
            self.listChanges.append("Deprecated API detected in line: " + node.lineno.__str__())
            self.listChanges.append("Content: \n" + unparse(node))
            api_object = self.list_completed_API[self.list_line_number.index(node.lineno)]
            if api_object["type"] == "ARGUMENT":
                actual_code = api_object["code"]
                new_parse = ast.parse(api_object["code"])
                call_node = new_parse.body[0].value
                original_code = unparse(call_node)
                self.positional_to_keyword(call_node)
                listScope = recurseScope(call_node)
                for n in listScope:
                    if isinstance(n, _ast.Call) and self.functionName[self.functionName.rfind(".") + 1:] in unparse(n):
                        # Check if the keyword is already present
                        isExist = False
                        listKeyword = node.keywords
                        for key in listKeyword:
                            if key.arg == self.parameterName:
                                isExist = True
                                break
                        if not isExist:
                            self.positional_to_keyword(n)
                updated_code = unparse(node).replace(original_code.rstrip(), unparse(call_node).rstrip()) + "\n"
                updated_node = ast.parse(updated_code.rstrip().lstrip())
                updated_node = updated_node.body[0].value
                self.listChanges.append("Updated content: \n" + unparse(updated_node))
                self.dict_change[node.lineno] = updated_node
                return updated_node
            else:
                self.positional_to_keyword(node)
                listScope = recurseScope(node)
                for n in listScope:
                    if isinstance(n, _ast.Call) and self.functionName[self.functionName.rfind(".") + 1:] in unparse(n):
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
                self.dict_change[node.lineno] = node
        return node

    def transform(self, tree):
        self.listChanges = []
        self.visit(tree)
        return self.dict_change

# Remove positional parameter
# TODO - NOT DONE YET
class PositionalParamRemover(ast.NodeTransformer):
    functionName = ""
    listChanges = []
    list_line_number = []
    list_completed_API = []

    def __init__(self, api_name, param_position, new_keyword, list_line_number, list_completed_API):
        self.functionName = api_name
        self.parameter_position = param_position
        self.parameter_keyword = new_keyword
        self.list_line_number = list_line_number
        self.dict_change = {}
        self.list_completed_API = list_completed_API
        super().__init__()

    def remove_positional_param(self, node: Call):
        # Function name is correct
        # PERHAPS FOR THIS WE NEED TO MAKE SURE THAT THE POSITIONAL ARGUMENT IS OF THE CORRECT TYPE
        # E.G. MAKE SURE THAT THE SECOND POSITIONAL ARGUMENT IS INTEGER TYPE
        # OR MAYBE WE ALSO NEED TO MAKE SURE THE NUMBER OF ARGUMENT IS CORRECT (E.G. 3 POSITIONAL PARAMETER)
        # THIS WILL BE A FUTURE TODO
        # CHECKING THE TYPE OF THE POSITIONAL ARGUMENT WILL BE HARDER SINCE IT CAN BE A NAME, API CALL, CONSTANT, ETC

        listPositionalParam = node.args
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



# The parameterValue that can be passed is limitted?
# E.g. String, Numeral, Boolean
class AddNewParameter(ast.NodeTransformer):
    functionName = ""
    parameterName = ""
    listChanges = []
    list_line_number = []
    dict_change = {}
    list_completed_API = []

    def __init__(self, api_name, parameter_name, parameter_type, parameter_value, list_line_number, list_completed_API):
        self.functionName = api_name
        self.parameterName = parameter_name
        self.parameterType = parameter_type
        self.parameterValue = parameter_value
        self.dict_change = {}
        self.list_completed_API = list_completed_API

        self.list_line_number = list_line_number
        super().__init__()

    def add_new_parameter(self, node: Call):
        if self.parameterType.lower() == "call":
            # API call or object instantiation
            try:
                keyword_value = ast.parse(self.parameterValue).body[0].value
            except Exception as e:
                exit(1)
        elif self.parameterType.lower() == "attribute":
            # class or object or package attribute type
            # new_value
            try:
                keyword_value = ast.parse(self.parameterValue).body[0].value
            except Exception as e:
                exit(1)
        elif self.parameterType.lower() == "name":
            # name or variable type
            try:
                keyword_value = ast.parse(self.parameterValue).body[0].value
            except Exception as e:
                exit(1)
        elif self.parameterType.lower() == "constant":
            # constant type
            keyword_value = ast.Constant(value=self.parameterValue, kind=None)
        else:
            print("Broken input")
        new_keyword = ast.keyword(arg=self.parameterName, value=keyword_value)
        node.keywords.append(new_keyword)
        return node

    def visit_Call(self, node: Call):
        if node.lineno in self.list_line_number:

            self.listChanges.append("Deprecated API detected in line: " + node.lineno.__str__())
            self.listChanges.append("Content: \n" + unparse(node))
            api_object = self.list_completed_API[self.list_line_number.index(node.lineno)]
            if api_object["type"] == "ARGUMENT":
                actual_code = api_object["code"]
                new_parse = ast.parse(api_object["code"])
                call_node = new_parse.body[0].value
                original_code = unparse(call_node)
                self.add_new_parameter(call_node)
                listScope = recurseScope(call_node)
                for n in listScope:
                    if isinstance(n, _ast.Call) and self.functionName[self.functionName.rfind(".") + 1:] in unparse(n):
                        # Check if the keyword is already present
                        isExist = False
                        listKeyword = node.keywords
                        for key in listKeyword:
                            if key.arg == self.parameterName:
                                isExist = True
                                break
                        if not isExist:
                            self.add_new_parameter(n)
                updated_code = unparse(node).replace(original_code.rstrip(), unparse(call_node).rstrip()) + "\n"
                updated_node = ast.parse(updated_code.rstrip().lstrip())
                updated_node = updated_node.body[0].value
                self.listChanges.append("Updated content: \n" + unparse(updated_node))
                self.dict_change[node.lineno] = updated_node
                return updated_node
            else:
                self.add_new_parameter(node)
                listScope = recurseScope(node)
                for n in listScope:
                    if isinstance(n, _ast.Call) and self.functionName[self.functionName.rfind(".") + 1:] in unparse(n):
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
                self.dict_change[node.lineno] = node
        return node

    def transform(self, tree):
        self.listChanges = []
        self.visit(tree)
        return self.dict_change


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
        self.dict_change = {}
        super().__init__()

    def default_value_transform(self, node: Call):
        nodeFuncName = getFunctionName(node)
        if nodeFuncName == self.functionName:
            # Function name is correct
            listKeywordParam = getKeywordArguments(node)
            isKeywordExist = False
            for keyword in listKeywordParam:
                if keyword.arg == self.parameterName:
                    isKeywordExist = True
            if not isKeywordExist:
                # If keyword is not exist yet, it means that the old API use the default value which is changed
                # in the new API. Therefore, we need to create a new node
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
        self.visit(tree)
