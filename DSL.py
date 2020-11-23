import ast
import sys
import os
from ast_transform_rule import *

# Tutorial based on this article:
# https://dbader.org/blog/writing-a-dsl-with-python

# 1. Change parameter name
# 2. Change positional parameter into keyword parameter
# 3. Remove parameter
# 4. Change method name
# 5. Complex change


# 1:
from input_utils import ApiSignature

"change_param_name fully_qualified_api_name filename old_param_name new_param_name "

# 2:
"positional_to_keyword fully_qualified_api_name filename parameter_position keyword_param_name "

# 3:
"remove_param fully_qualified_api_name filename keyword_param_name "
"remove_param fully_qualified_api_name filename parameter_position "

# 4:
"change_method_name old_fully_qualified_name filename new_fully_qualified_name "

# 5:
"complex_transform api_name filename example_filename"

# dsl1.py


# Input: List of DSL commands, filename to be processed with the commands, and api_name
# Output: Tree, List Edited Lines
def run_DSL(list_DSL, filename, api_signature: ApiSignature):
    original_api_name = api_signature.api_name
    with open(filename, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read())
        api_name = api_signature.api_name
        parameter_name = ""
        list_edited_line = {}
        positional_skip = 0
        hasConstraint = False
        for dsl in list_DSL:
            # Positional skip is used to mitigate the effect of the previous-existing positional param being removed
            # For example, due to positional_to_keyword or positional_param_remove
            dsl = dsl.lstrip().rstrip()
            list_completed_api = get_list_API(tree, api_signature)
            list_line_number = get_list_line_number(list_completed_api)
            splitted_dsl = dsl.split(" ")
            if splitted_dsl[0] == "rename_method":
                new_name = splitted_dsl[3]
                nameTransformer = ApiNameTransformer(api_name, new_name, list_line_number, list_completed_api)
                dict_change = nameTransformer.transform(tree)
                for key, value in dict_change.items():
                    list_edited_line[key] = value
                # Convert the API name into the new name for future detection
                api_name = new_name
                api_signature.api_name = new_name
            elif splitted_dsl[0] == "add_parameter":
                param_name = splitted_dsl[1]
                param_value = splitted_dsl[3]
                # currently default to call until there is new way to detect the type
                param_type = "call"
                addParamTransformer = AddNewParameter(api_name, param_name, param_type, param_value, list_line_number, list_completed_api)
                dict_change = addParamTransformer.transform(tree)
                for key, value in dict_change.items():
                    list_edited_line[key] = value
            elif splitted_dsl[0] == "rename_parameter":
                old_param_name = splitted_dsl[1]
                new_param_name = splitted_dsl[3]
                # currently default to call until there is new way to detect the type
                renameParamTransformer = KeywordParamChanger(api_name, old_param_name, new_param_name, list_line_number, list_completed_api)
                dict_change = renameParamTransformer.transform(tree)
                for key, value in dict_change.items():
                    list_edited_line[key] = value
            elif splitted_dsl[0] == "positional_to_keyword":
                param_position = splitted_dsl[2]
                param_keyword = splitted_dsl[4]
                positionalToKeywordTransformer = PositionalToKeyword(api_name, int(param_position) - positional_skip, param_keyword, list_line_number, list_completed_api)
                dict_change = positionalToKeywordTransformer.transform(tree)
                positional_skip += 1
                for key, value in dict_change.items():
                    list_edited_line[key] = value
            elif splitted_dsl[0] == "remove_parameter":
                deleted_keyword = splitted_dsl[1]
                keywordRemover = KeywordParamRemover(api_name, deleted_keyword, list_line_number, list_completed_api)
                dict_change = keywordRemover.transform(tree)
                for key, value in dict_change.items():
                    list_edited_line[key] = value
            elif splitted_dsl[0] == "remove_api":
                apiRemover = RemoveAPI(api_name, list_line_number, list_completed_api)
                dict_change = apiRemover.transform(tree)
                for key, value in dict_change.items():
                    list_edited_line[key] = value
            # Process the constraint here
            ifIndex = -1
            for index, element in enumerate(splitted_dsl):
                if element.lower() == "IF".lower():
                    hasConstraint = True
                    ifIndex = index
                    break
            # Need to add value constraint here
            code_string = ""
            if hasConstraint:
                # Check if the constraint has not
                hasNot = False
                if splitted_dsl[ifIndex + 1].lower() == "NOT".lower():
                    hasNot = True
                    ifIndex = ifIndex + 1
                parameter_name = splitted_dsl[ifIndex + 1]
                constraint_String = splitted_dsl[ifIndex + 2] + " " + splitted_dsl[ifIndex + 3]
                not_string = ""
                if hasNot:
                    not_string = "not "
                if constraint_String.lower() == "HAS TYPE".lower():
                    type_string = splitted_dsl[ifIndex + 4]
                    code_string = "if "+ not_string + "isinstance(" + "TEMPORARY_PARAMETER_NAME" + ", " + type_string + "):"
                elif constraint_String.lower() == "HAS VALUE".lower():
                    # value_string = splitted_dsl[ifIndex + 4]
                    value_string = ""
                    for i in range(ifIndex + 4, len(splitted_dsl)):
                        value_string = value_string + " " + splitted_dsl[i]
                    code_string = "if " + not_string + "TEMPORARY_PARAMETER_NAME" + value_string + ":"

        api_signature.api_name = original_api_name
        return tree, list_edited_line, hasConstraint, code_string, parameter_name

# Input: modified tree and list diff from the Run_DSL function and the filename to be changed
# Output: Dictionary with key being the position of the diff and the value being a
# tuples of the old value and list of the new value
def get_list_diff(tree, list_diff, filename):
    list_position = list(list_diff.keys())
    try:
        list_position.remove(0)
    except:
        pass
    old_tree = ast.parse(open(filename, encoding="utf-8").read())

    with open("old_file.py", "w", encoding="utf-8") as old_open:
        old_open.write(unparse(old_tree))

    with open("new_file.py", "w", encoding="utf-8") as new_open:
        new_open.write(unparse(tree))

    old_list = unparse(old_tree).split("\n")
    new_list = unparse(tree).split("\n")
    diff_dict = {}
    i = 0
    position = 1
    while i < len(old_list):
        if old_list[i] == new_list[i]:
            old_list.pop(i)
            new_list.pop(i)
            position += 1
        elif new_list[i][0:4] == "from":
            # Special case if import statement in the new key
            diff_dict[1] = "", [new_list.pop(i)]
            position += 1
        else:
            old_key = old_list.pop(i)
            new_key = [new_list.pop(i)]
            try:
                actual_position = list_position.pop(0)
            except:
                actual_position = 1
            # while len(list_position) > 0 and list_position[0] < position:
            #     actual_position = list_position.pop(0)
            if "EMPTYSHOULDBEDELETED" in new_key.__str__():
                diff_dict[actual_position] = old_key, ["pass"]
            else:
                diff_dict[actual_position] = old_key, new_key
            position += 1
    os.remove("old_file.py")
    os.remove("new_file.py")
    return diff_dict
